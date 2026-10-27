"""Root's concrete review receipt for exactly one separately pinned trial."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / 'artifacts/runs/agentic/phrase_program_execution_plan_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads((PLAN / 'protocol.json').read_text())
    if sha(PLAN / 'protocol.json') != 'b6a164f42c8f7bfe62a3a814dc1aba7ddf3e88278ec52f51b13817619270dcad':
        raise ValueError('reviewed protocol changed')
    if len(protocol['pins']) != 325:
        raise ValueError('expected exact reviewed inventory')
    for path, digest in protocol['pins'].items():
        if sha(path) != digest:
            raise ValueError('reviewed source changed: ' + path)
    reviewed = ROOT / 'artifacts/runs/agentic/phrase_program_review_v1/final_protocol_review.json'
    review = json.loads(reviewed.read_text())
    # Its independently reviewed digest must appear in the actual JSON receipt.
    if 'b6a164f42c8f7bfe62a3a814dc1aba7ddf3e88278ec52f51b13817619270dcad' not in json.dumps(review):
        raise ValueError('independent protocol review mismatch')
    executor = ROOT / 'experiments/agentic/phrase/execute_program_v1.py'
    if sha(executor) != '68ebe4f3dcf1457a5cddcb4db6b5099ec10db045e47fe49aa4ee18a5149cbea5':
        raise ValueError('reviewed executor changed')
    if Path(protocol['output']).exists():
        raise ValueError('one-trial budget already consumed')
    value = {'approved': True, 'authority': 'root, within user-authorized private local development',
             'protocol_sha256': sha(PLAN / 'protocol.json'), 'executor_sha256': sha(executor),
             'independent_review_sha256': sha(reviewed),
             'expression_analysis_plan_sha256': sha(PLAN / 'expression_analysis_plan.json'),
             'maximum_trials': 1, 'maximum_online_calls': 400, 'maximum_warmups': 1,
             'maximum_physics_steps': 8000, 'retry_allowed': False,
             'assessment': 'Finite common-only program passes strict references/continuous ranges/reset/actual eight-feed18row checks; direct frozen upstream numeric loop and exact in-loop prefix still mandatory.',
             'limitations': ['One physical feasibility case, no app/catalog/training/hardware admission.',
                             'Offline music features informed a real model proposal; fixed clock does not synchronize beats.',
                             'Physical screens measure minimum height/joint tracking, not human preference or recovery.',
                             'Gains and heading both differ from earlier fixed composition; no causal isolation.',
                             'Prior model attempt failed/completion unconfirmed and failed relative-path preparation remain preserved.'],
             'new_physics_steps_at_review': 0, 'new_onnx_calls_at_review': 0}
    with (PLAN / 'root_execution_review_v1.json').open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'approved': True, 'review_sha256': sha(PLAN / 'root_execution_review_v1.json')}))


if __name__ == '__main__':
    main()
