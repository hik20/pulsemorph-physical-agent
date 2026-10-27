"""Render blind A/B cases with mj_forward only; the positive control is kinematic."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/runs/agentic/app_vision_probe_v1"
MEDIA = ROOT / "artifacts/media/agentic/app_vision_probe_v1"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    job = ROOT / "artifacts/runs/agentic/app/jobs/20261027T023422_8efe4bc4"
    dirs = [job / "trials/baseline", job / "trials/trial_02"]
    sources, arrays = [], []
    xml = ROOT / "research/vendor/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
    for directory in dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        for name in ("qpos.npy", "state_times.npy"):
            if sha(directory / name) != manifest["artifact_hashes"][name]: raise ValueError("Recorded physical trace artifact hash mismatch")
        if manifest["status"] != "success" or manifest["actual_physics_steps"] != 4000:
            raise ValueError("Expected actual archived four-second physical trial")
        if manifest["preexecution_input_hashes"][str(xml.relative_to(ROOT))] != sha(xml): raise ValueError("Render body XML differs")
        qpos, times = np.load(directory / "qpos.npy", allow_pickle=False), np.load(directory / "state_times.npy", allow_pickle=False)
        indices = [int(np.argmin(abs(times - time))) for time in (.8, 1.6, 2.4, 3.2)]
        if not np.allclose(times[indices], [.8, 1.6, 2.4, 3.2], rtol=0, atol=1e-8): raise ValueError("Requested frame times absent")
        if qpos.shape[1] != 36 or not np.isfinite(qpos).all(): raise ValueError("Recorded qpos invalid")
        arrays.append(qpos[indices].copy())
        sources.append({"directory": directory.relative_to(ROOT).as_posix(), "manifest_sha256": sha(directory / "manifest.json"),
                        "qpos_sha256": sha(directory / "qpos.npy"), "times_sha256": sha(directory / "state_times.npy"),
                        "selected_indices": indices, "selected_times_s": times[indices].tolist()})
    document = ET.parse(xml); doc = document.getroot(); compiler = doc.find("compiler")
    meshdir = (xml.parent / compiler.get("meshdir", "")).resolve()
    mesh_hashes = {}
    for mesh in doc.findall("./asset/mesh"):
        path = meshdir / mesh.attrib["file"]
        for directory in dirs:
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["preexecution_input_hashes"].get(str(path.relative_to(ROOT))) != sha(path): raise ValueError("Render mesh differs from recorded input")
        mesh_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    compiler.set("meshdir", str(meshdir))
    for sensor in doc.findall("sensor"): doc.remove(sensor)
    world = doc.find("worldbody")
    if not any(g.get("type") == "plane" for g in world.findall("geom")):
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .05", rgba=".12 .16 .22 1")
    if not world.findall("light"):
        ET.SubElement(world, "light", pos="2 0 5", dir="0 0 -1", diffuse=".6 .6 .6", directional="true")
    model_path = MEDIA / "render_model.xml"
    if model_path.exists(): raise FileExistsError("Blind rendering already prepared")
    document.write(model_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(model_path)); data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [.5, .5, .5]; model.vis.headlight.diffuse[:] = [.8, .8, .8]
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid, camera.distance, camera.azimuth, camera.elevation = 1, 2.8, 140, -10
    renderer = mujoco.Renderer(model, height=360, width=320)
    def panel(state):
        data.qpos[:] = state; data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return Image.fromarray(renderer.render()).copy()
    posed = arrays[0].copy()
    changes = {"left_shoulder_roll_joint": 1.4, "right_shoulder_roll_joint": -1.4,
               "left_elbow_joint": 1.1, "right_elbow_joint": 1.1}
    for name, value in changes.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or not model.jnt_limited[joint] or not model.jnt_range[joint, 0] <= value <= model.jnt_range[joint, 1]:
            raise ValueError("Positive-control pose does not fit explicit joint mapping/ranges")
        posed[:, model.jnt_qposadr[joint]] = value
    np.save(MEDIA / "kinematic_positive_only_qpos.npy", posed)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    cases = []
    try:
        first = [panel(state) for state in arrays[0]]
        real = [panel(state) for state in arrays[1]]
        positive = [panel(state) for state in posed]
        for case_id, kind, b in [("v01", "null_duplicate", [p.copy() for p in first]), ("v02", "kinematic_positive_only", positive), ("v03", "archived_physical_edit", real)]:
            canvas = Image.new("RGB", (1320, 760), "#101820"); draw = ImageDraw.Draw(canvas)
            draw.text((8, 195), "A", font=font, fill="white"); draw.text((8, 555), "B", font=font, fill="white")
            equivalence = []
            for index, (a, image_b) in enumerate(zip(first, b)):
                draw.text((45 + index * 320, 5), f"{index + 1}   {(.8, 1.6, 2.4, 3.2)[index]:.1f}s", font=font, fill="white")
                canvas.paste(a, (40 + index * 320, 40)); canvas.paste(image_b, (40 + index * 320, 400))
                equivalence.append(a.tobytes() == image_b.tobytes())
            if kind == "null_duplicate" and not all(equivalence): raise ValueError("Null body panels must be pixel-identical")
            path = MEDIA / (case_id + ".png"); canvas.save(path)
            cases.append({"case_id": case_id, "kind_hidden_from_prompt": kind, "image_path": path.relative_to(ROOT).as_posix(), "image_sha256": sha(path),
                          "body_panel_A_B_pixel_equal": equivalence, "B_physical_execution": kind != "kinematic_positive_only"})
    finally: renderer.close()
    receipt = {"schema": "pulsemorph.blind_vision_render.v1", "new_physics_steps": 0, "new_ONNX_calls": 0, "new_model_calls": 0,
               "operation": "mj_forward on recorded qpos; positivecontrol modified qpos is kinematic-only and was never physically executed",
               "sources": sources, "body_xml_sha256": sha(xml), "render_model_sha256": sha(model_path), "mesh_hashes": mesh_hashes,
               "positive_kinematic_changes": changes, "positive_qpos_sha256": sha(MEDIA / "kinematic_positive_only_qpos.npy"),
               "camera": {"tracking_body_id": 1, "distance": 2.8, "azimuth": 140, "elevation": -10},
               "cases": cases, "renderer_script_sha256": sha(Path(__file__)), "blindness": "Only A/B and frame/time indices shown. No condition/gain/pass/source labels."}
    with (OUT / "render_manifest.json").open("x", encoding="utf-8") as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps({"blind_cases": 3, "null_panels_pixel_equal": True, "new_physics": 0, "new_models": 0}))


if __name__ == "__main__": main()
