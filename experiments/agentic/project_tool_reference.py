"""Metadata-only compatible export; raw recordings retain their admission failure.

No physics, policy execution, training admission, or trial mutation. Only the
two known redundant top-level fields may be removed; nested producer remains.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pulsemorph.evidence import sha256, write_new_json
from pulsemorph.motion.contracts import MotionReference, _plain, load_reference, save_reference
from pulsemorph.motion.g1 import assemble_reference_inputs, sample_reference, FUTURE_OFFSETS_S

ARRAYS = ('time_offsets_s', 'joint_position_rad', 'joint_velocity_rad_s', 'anchor_orientation_xyzw')
ALLOWED = {'schema_version', 'reference_id', 'producer', 'body', 'controller', 'frame', 'context', 'validity', 'observation', 'provenance'}
TOOLS = ('global_clock_v1', 'local_clock_1p4_2p8_v1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inventory(path):
    return {str(p.relative_to(path)): sha256(p) for p in path.rglob('*') if p.is_file()}


def array_proof(original, projected):
    proof = {}
    for name in ARRAYS:
        a, b = getattr(original, name), getattr(projected, name)
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('numeric payload changed: ' + name)
        import hashlib
        proof[name] = dict(dtype=str(a.dtype), shape=list(a.shape), payload_sha256=hashlib.sha256(a.tobytes()).hexdigest(), unchanged=True)
    return proof


def project_reference(original: MotionReference, manifest: dict):
    """Return (new reference, disclosure), verifying strict rejection and repair.

    ``manifest`` is the original execution manifest, not the reference manifest.
    Caller remains responsible for binding it to the recorded protocol/proposal.
    Exactly two unsupported fields must exist; no unknown field is discarded.
    """
    metadata = _plain(original.metadata)
    if set(metadata) - ALLOWED != {'control_tool', 'experiment_kind'}:
        raise ValueError('expected exactly two known unsupported top-level fields')
    tool = metadata['control_tool']
    if (tool not in TOOLS or tool != metadata['producer'].get('control_tool')
            or tool != manifest.get('control_tool')
            or metadata['producer'] != manifest.get('producer')
            or metadata['experiment_kind'] != 'tool_selection_v1'
            or manifest.get('experiment_kind') != 'tool_selection_v1'):
        raise ValueError('metadata/manifest producer or experiment conflict')
    try:
        assemble_reference_inputs(original, 0.)
    except ValueError as error:
        if str(error) != 'unsupported target metadata fields':
            raise ValueError('raw rejection is not the known top-level issue') from error
        raw_error = str(error)
    else:
        raise ValueError('raw reference unexpectedly passes strict admission')
    removed = {key: metadata.pop(key) for key in ('control_tool', 'experiment_kind')}
    projected = replace(original, metadata=metadata)
    proof = array_proof(original, projected)
    for offset in (0., 3.98):
        actual = assemble_reference_inputs(projected, offset)
        sampled = sample_reference(original, offset + FUTURE_OFFSETS_S)
        expected = dict(mimic_future_dof_pos=sampled['dof_pos'][None].astype('float32'),
                        mimic_future_dof_vel=sampled['dof_vel'][None].astype('float32'),
                        mimic_future_anchor_rot=sampled['anchor_rot'][None].astype('float32'))
        if any(actual[key].tobytes() != expected[key].tobytes() for key in expected):
            raise ValueError('projected target differs from original numeric sampling')
    return projected, dict(removed_top_level_fields=removed, retained_nested_producer=metadata['producer'],
        original_strict_g1_admission=False, original_strict_g1_rejection=raw_error,
        projection_strict_g1_admission=True, strict_probe_offsets_s=[0., 3.98], arrays=proof,
        numerical_only_no_resimulation=True, training_admission=False)


def export(pilot, output, failed_audit):
    """Export exactly 100 existing trial references into a fresh sibling tree."""
    pilot, output, failed_audit = map(lambda p: Path(p).resolve(), (pilot, output, failed_audit))
    if output.exists() or output == pilot or pilot in output.parents or output in pilot.parents:
        raise ValueError('fresh output outside original evidence required')
    audit = read(failed_audit)
    if audit.get('audit_passed') is not False:
        raise ValueError('explicit failed audit required')
    before = inventory(pilot)
    dependencies = [Path(__file__).resolve()] + [ROOT / 'src/pulsemorph' / name for name in
        ('evidence.py', 'motion/contracts.py', 'motion/g1.py')]
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in dependencies}
    protocol_path = pilot / 'protocol.json'
    protocol = read(protocol_path)
    if protocol.get('schema') != 'pulsemorph.agentic_tool_selection_protocol.v2':
        raise ValueError('v2 source protocol required')
    trials = sorted(p for p in (pilot / 'trials').iterdir() if p.is_dir())
    if len(trials) != 100:
        raise ValueError('exactly 100 recorded trials required')
    output.mkdir(parents=True, exist_ok=False)
    for relative, digest in pins.items():
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if sha256(destination) != digest:
            raise ValueError('exporter/helper snapshot changed')
    rows = []
    audit_digest = sha256(failed_audit)
    for trial in trials:
        manifest_path, proposal_path = trial / 'manifest.json', trial / 'proposal.json'
        manifest, proposal = read(manifest_path), read(proposal_path)
        original_path = trial / 'common_reference/manifest.json'
        original_arrays = original_path.with_name('arrays.npz')
        if (manifest.get('status') != 'success' or manifest.get('candidate_id') != trial.name
                or proposal.get('candidate_id') != trial.name
                or proposal.get('protocol_sha256') != sha256(protocol_path)
                or manifest.get('proposal_sha256') != sha256(proposal_path)
                or manifest.get('control_tool') != proposal.get('control_tool')):
            raise ValueError('recorded execution binding mismatch: ' + trial.name)
        artifact_hashes = {name.replace('\\', '/'): value for name, value in manifest['artifact_hashes'].items()}
        for name in ('common_reference/manifest.json', 'common_reference/arrays.npz'):
            if artifact_hashes.get(name) != sha256(trial / name):
                raise ValueError('original reference artifact changed')
        if sha256(trial / 'source.motion') != manifest['motion_sha256']:
            raise ValueError('original source motion changed')
        original = load_reference(original_path)
        projected, disclosure = project_reference(original, manifest)
        destination = output / 'references' / trial.name
        saved_manifest = save_reference(projected, destination)
        reloaded = load_reference(saved_manifest)
        array_proof(original, reloaded)
        # Require NPZ payload dtypes/shape/bytes too, not just loader conversion.
        with np.load(original_arrays, allow_pickle=False) as a, np.load(destination / 'arrays.npz', allow_pickle=False) as b:
            if set(a.files) != set(ARRAYS) or set(b.files) != set(ARRAYS):
                raise ValueError('archive key inventory changed')
            for name in ARRAYS:
                if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape or a[name].tobytes() != b[name].tobytes():
                    raise ValueError('archive numeric payload changed')
        rows.append(dict(candidate_id=trial.name, original_manifest=dict(path=str(original_path), sha256=sha256(original_path)),
            original_arrays=dict(path=str(original_arrays), sha256=sha256(original_arrays)),
            trial_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
            proposal=dict(path=str(proposal_path), sha256=sha256(proposal_path)),
            protocol_sha256=sha256(protocol_path), source_motion=manifest['source_motion'], source_motion_sha256=manifest['motion_sha256'],
            projected_manifest=dict(path=str(saved_manifest), sha256=sha256(saved_manifest)),
            projected_arrays=dict(path=str(destination / 'arrays.npz'), sha256=sha256(destination / 'arrays.npz')), disclosure=disclosure))
    for relative, digest in pins.items():
        source = ROOT / relative
        if sha256(source) != digest:
            raise ValueError('exporter/helper changed during projection')
        if sha256(output / 'source_snapshot' / relative) != digest:
            raise ValueError('exporter/helper snapshot changed during projection')
    after = inventory(pilot)
    if after != before or sha256(failed_audit) != audit_digest:
        raise ValueError('original evidence or failed audit changed during projection')
    index = dict(schema='pulsemorph.agentic_tool_reference_projection.v1', pilot=str(pilot),
        failed_audit=dict(path=str(failed_audit), sha256=audit_digest),
        original_strict_g1_admission=False, projection_strict_g1_admission=True,
        numerical_only_no_resimulation=True, reference_count=len(rows), references=rows,
        source_hashes_before=pins, source_hashes_after={name: sha256(ROOT / name) for name in pins},
        original_inventory_before=before, original_inventory_after=after, original_evidence_unchanged=True)
    write_new_json(output / 'index.json', index)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'output', 'failed-audit'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = export(args.pilot, args.output, args.failed_audit)
    print(json.dumps(dict(reference_count=result['reference_count'], output=str(args.output))))
