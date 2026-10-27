# Separate code, model, body and media dependencies

Tested prepared host: Windows, CPython 3.11.4; CPU ONNX execution. The package records direct numerical versions in `requirements-observed.txt`. The scripts do not install or purchase services automatically.

| Component | Role | Included here |
|---|---|---|
| Own PulseMorph agentic source/tests | Model orchestration, compilation, local apps, experimental control | Source only |
| [MuJoCo](https://mujoco.org/) 3.3.7 | Actual rigid-body/contact dynamics | Python dependency, not bundled runtime |
| [NVIDIA ProtoMotions](https://github.com/NVlabs/ProtoMotions) | G1 tracking interfaces and pretrained motor foundation | Not vendored in archive |
| G1 MJCF, meshes and controller YAML | Body geometry, inertia, joint/force limits | Not redistributed |
| G1 pretrained compiled ONNX actor | Low-level tracking; diagnostic-only in some later controller experiments | Not redistributed |
| Safe initial-carrier NPZ/profile | Known initial body pose and provenance manifest | Private prepared-host prerequisite |
| Meterless upstream WorldModel | Existing local evidence memory | External dependency, exact commit in evidence |
| NVIDIA Nemotron / OpenRouter | Optional text/visual reasoning service | Adapter code only; private key required |
| Procedural music and simulation recordings | Demo media | Separate files, not source archive |

Required model path relative to upstream source is `data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`, with its matching `.yaml`. The body is `protomotions/data/assets/mjcf/g1_holo_compat.xml` and the G1 mesh closure. Use the exact pins in `asset_identity.json`, not an arbitrary model or silently changed body. Review the upstream model cards/licenses and asset provenance separately. A source-code license does not automatically license checkpoints, source motions or meshes.

The profile path used in the recorded host is a dedicated private cache, not an asset distributed by this archive. The source checks its hashes and refuses changed prerequisites. A complete new-machine asset acquisition recipe is still open; do not describe these setup commands as a verified clean installation.

External reference methods used in design include [SayTap](https://saytap.github.io/) and prior LLM robotics. Their existence is why this demo does not claim to be the first use of LLMs for robotics. Their results are not PulseMorph results.
