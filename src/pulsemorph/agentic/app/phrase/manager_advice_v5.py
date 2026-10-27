"""Add conditional reference advice after rejection, without editing controls."""
import copy
import hashlib
import json
import time
from .manager_goals_v4 import PhraseGoalsManager
from .manager import write, json_bytes, PhraseCancelled
from .gain_advice_v1 import advise_gain_interval, AdviceTimeout, AdviceCancelled
from ...phrase_program import ProgramCompileError, ARM_PARTS, parse_program

MAX_ADVICE_FIELDS = 2
ADVICE_DEADLINE_S = 45.
PER_FIELD_DEADLINE_S = 20.


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')


def _fields_for_report(report):
    """Only arm gains can change these named arm-joint constraints."""
    violations = report.get('violations', [])
    if not isinstance(violations, list):
        return []
    fields = []
    for side in ('left', 'right'):
        if any(side + '_' + part in violations for part in ARM_PARTS):
            fields.extend(('second_' + side + '_gain', 'first_' + side + '_gain'))
    return fields[:MAX_ADVICE_FIELDS]


class PhraseAdviceManager(PhraseGoalsManager):
    def _advice(self, identity, program, index, report):
        program = parse_program(program)
        started = time.monotonic()
        deadline = started + ADVICE_DEADLINE_S
        fields = _fields_for_report(report)
        self._check(identity)
        self._update(identity, phase=f'Aday {index+1}: koşullu referans sınırları inceleniyor')
        results = []
        for field in fields:
            self._check(identity)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                results.append({'field': field, 'status': 'unavailable', 'reason': 'batch_deadline'})
                break
            try:
                evidence = advise_gain_interval(self.runtime.assets, program, field,
                    cancel_event=self._cancel[identity], timeout_s=min(remaining, PER_FIELD_DEADLINE_S))
            except AdviceTimeout:
                evidence = {'field': field, 'status': 'unavailable', 'reason': 'cooperative_deadline'}
            # Invalid inputs, profile/integrity errors and cancellation propagate.
            results.append(evidence)
        self._check(identity)
        guard = getattr(self.runtime, '_unchanged', None)
        if callable(guard):
            guard()
        return {'schema': 'pulsemorph.phrase_conditional_gain_advice_batch.v1',
            'status': 'completed' if results else 'unavailable',
            'program_sha256': hashlib.sha256(_canonical(program)).hexdigest(),
            'fields_requested': fields, 'intervals': results,
            'max_fields': MAX_ADVICE_FIELDS,
            'max_reference_compile_attempts': 4 * len(fields),
            'max_reference_future_queries': 1600 * len(fields),
            'cooperative_deadline_s': ADVICE_DEADLINE_S,
            'elapsed_s': time.monotonic() - started,
            'execution_authority': False,
            'qualification': 'Each numerical interval holds every other gain, order, heading and asset fixed. Individual intervals are not a simultaneous feasible region. No control or target is changed, no physics feasibility is implied, and the original compiler must validate each proposal. Deadline checks occur between bounded compiler calls and polynomial intervals.'}

    def _run_trial(self, identity, program, index, music):
        try:
            return super()._run_trial(identity, program, index, music)
        except ProgramCompileError as error:
            if index < 0:
                raise
            original = copy.deepcopy(error.report)
            relative = f'advice/trial_{index:02d}.json'
            path = self.jobs_dir / identity / relative
            attempt_path = path.with_suffix('.attempt.json')
            # A cancellation/timeout/integrity failure must not erase the real
            # rejection that happened before advisory work started.
            write(attempt_path, {'schema': 'pulsemorph.phrase_advice_attempt.v1',
                'job_id': identity, 'candidate_index': index,
                'completion_unconfirmed': True,
                'program': parse_program(program),
                'original_compile_report': original,
                'new_model_ONNX_FK_physics_calls': 0}, exclusive=True)
            try:
                evidence = self._advice(identity, program, index, original)
            except (AdviceCancelled, PhraseCancelled):
                self._update(identity, active_candidate=None)
                raise PhraseCancelled('cancelled during conditional reference advice')
            receipt = {'schema': 'pulsemorph.phrase_advice_receipt.v1',
                'job_id': identity, 'candidate_index': index,
                'program': parse_program(program),
                'original_compile_report_sha256': hashlib.sha256(json_bytes(original)).hexdigest(),
                'attempt_receipt_sha256': hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                'conditional_gain_advice': evidence,
                'new_model_ONNX_FK_physics_calls': 0}
            write(path, receipt, exclusive=True)
            self._check(identity)
            report = copy.deepcopy(original)
            report['conditional_gain_advice'] = dict(evidence,
                saved_receipt=relative,
                saved_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            raise ProgramCompileError(str(error), report) from error

    def _model(self, identity, purpose, prompt, schema):
        if purpose.startswith('feedback_'):
            prompt = ('LOCAL ADVISORY TOOL: The recorded rejection may include conditional_gain_advice. '
                'It is numerical reference evidence, not execution authority or a physical-success bound. '
                'Each field interval fixes ALL its recorded other gains, order, heading and assets. '
                'Do not combine separate intervals as a simultaneous feasible region. Choose controls yourself; '
                'there is no automatic clamp or replacement. Every new program still goes through the unchanged '
                'compiler and physical checks. Keep the original fixed goals and original candidate/model budget. '
                'An unavailable interval is unknown, not permission or a zero interval.\n' + prompt)
        # Initial prompt remains byte-bound to prepared_plan_v4.json.
        return super()._model(identity, purpose, prompt, schema)
