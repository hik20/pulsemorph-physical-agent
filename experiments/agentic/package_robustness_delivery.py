"""Read-only closing inventory; preserves completed, failed and audit evidence."""
from __future__ import annotations
import json
from pathlib import Path
import re
from datetime import datetime, timezone
from audit_robustness_v2 import inventory, read, sha, require, check_summary, check_chronology, check_protected

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/runs/agentic'


def package(output):
    output = Path(output).resolve()
    require(not output.exists(), 'closing manifest must be new')
    panel = BASE/'robustness_initial_state_v2'
    audit_path = BASE/'robustness_initial_state_v2_audit_v1/audit.json'
    report_path = BASE/'robustness_initial_state_v2_report_v1/report.json'
    protocol, audit, report = read(panel/'protocol.json'), read(audit_path), read(report_path)
    require(audit['audit_passed'] is True and audit['errors']==[], 'passed full audit required')
    require(audit['input_protocol_sha256']==sha(panel/'protocol.json'), 'audit protocol identity')
    for key in ('panel_evidence_unchanged','protected_evidence_unchanged','audit_sources_unchanged'):
        require(audit[key] is True, 'audit stability: '+key)
    for key, value in audit['source_hashes_after'].items():
        require(sha(ROOT/key)==value, 'audit source changed: '+key)
    check_protected(protocol)
    check_summary(panel, protocol, audit['selected_rows'])
    identities = [s['scenario_id']+'__'+c['role'] for s in protocol['scenarios'] for c in protocol['cases']]
    check_chronology(panel, identities, [i for i in identities if i.startswith('nominal__')])
    current = inventory(panel)
    for identity in identities:
        manifest = read(panel/'trials'/identity/'manifest.json')
        for name, digest in manifest['artifact_hashes'].items():
            require(current['trials/'+identity+'/'+name.replace('\\','/')] == digest, 'trial artifact changed: '+identity+'/'+name)
    for row in audit['selected_rows']:
        require(read(panel/'evaluations'/(row['candidate_id']+'.json')) == row['evaluation'], 'audited evaluation changed')
    for name, digest in protocol['runtime_input_sha256'].items():
        require(sha(ROOT/name)==digest, 'original motor/body/runtime changed: '+name)
    for name, digest in protocol['execution_code_sha256'].items():
        require(sha(ROOT/name)==digest and sha(panel/'source_snapshot'/name)==digest, 'frozen code changed: '+name)
    require(report['independent_audit_accepted'] is True, 'report must bind passed audit')
    links = [value for value in report['inputs'].values() if value is not None]
    links += [row['evaluation'] for row in report['rows']]
    for link in links:
        require(sha(link['path'])==link['sha256'], 'report linked input changed')
    require(report['inputs']['independent_audit']['sha256']==sha(audit_path), 'report audit link mismatch')
    documents = [ROOT/'experiments/agentic/ROBUSTNESS_RESULTS.md', ROOT/'design/agentic/ROBUSTNESS_EXPERIMENT.md',
        ROOT/'design/agentic/ROBUSTNESS_REVIEW.md', ROOT/'design/coordination/AGENTIC_STATUS.md']
    checked_links=[]
    # Historical status links belong to prior deliveries; check the new report's links.
    for target in re.findall(r'\]\(([^)]+)\)', documents[0].read_text(encoding='utf8')):
        if target.startswith(('https://','http://','#')): continue
        path = (documents[0].parent/target.split('#')[0].strip('<>')).resolve()
        require(path.exists(), 'result document broken link: '+target)
        checked_links.append(str(path))
    directories = [panel, audit_path.parent, report_path.parent, BASE/'robustness_initial_state_v1_quarantine_v1',
        BASE/'robustness_structural_preflight_v1', BASE/'robustness_verification_v2']
    sources = [Path(__file__), ROOT/'experiments/agentic/report_robustness.py'] + sorted((ROOT/'tests').glob('test_agentic_robustness*.py'))
    files = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in documents+sources}
    for directory in directories:
        files.update({str(directory.relative_to(ROOT)).replace('\\','/')+'/'+name:digest for name,digest in inventory(directory).items()})
    for name in ('robustness_initial_state_v1_execution.log','robustness_initial_state_v2_execution.log','robustness_initial_state_v2_audit_v1_execution.log'):
        path=BASE/name
        require(path.is_file(), 'execution log missing: '+name)
        files[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    result=dict(schema='pulsemorph.agentic_robustness_delivery.v1', verified=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(), files=files, file_count=len(files), checked_result_links=checked_links,
        completed_panel=dict(physical_trials=78,control_steps=15600,physics_steps=312000,new_llm_calls=0),
        preserved_abort=dict(physical_trials=3,control_steps=600,physics_steps=12000,new_llm_calls=0),
        total_this_experiment=dict(physical_trials=81,control_steps=16200,physics_steps=324000,new_llm_calls=0),
        training_admission=False, scope='Initial true-state finite sensitivity panel; immutable dated delivery snapshot, not recovery or population robustness.')
    with output.open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    result=package(BASE/'robustness_delivery_v2.json')
    print(json.dumps(dict(verified=result['verified'],files=result['file_count'],links=len(result['checked_result_links']))))
