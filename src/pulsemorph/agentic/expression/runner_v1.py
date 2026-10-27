"""Private, bounded music/instruction → expression → physical-feedback jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import uuid
import zipfile

from ..app.integrations import AppIntegrations
from ..app.phrase.assets import sha
from ..app.phrase.motor_response_v1 import bridge
from ..phrase_expression_v1 import describe_bases, GAIN_FIELDS, parse_expression
from ..phrase_program import ProgramCompileError
from .goals_v1 import parse_goals, evaluate_goals, select_final
from .planner_v1 import plan_expression
from .render_v1 import render_comparison

TERMINAL = {'completed', 'unsupported', 'failed', 'cancelled'}
SOURCE_ID = 'expression_high5_left_components_v1'
CACHE = Path.home() / '.cache/pulsemorph-agentic'
JOB_ROOT = CACHE / 'expression-studio-v1/jobs'
HISTORICAL_UNIT = CACHE / 'analysis/expression-motor-v1/prepared_v2/trials/unit'
PILOT = CACHE / 'analysis/expression-motor-v1/prepared_v2/completed.json'
PILOT_SHA = '488af8ec96318572029481c9c7f69e30b7b5dd7b217c8562fb33fffd85888772'
MAX_MODELS, MAX_CANDIDATES, WALL_SECONDS = 3, 3, 1800


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf8')


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def safe_directory(path, allowed):
    path, allowed = Path(path).absolute(), Path(allowed).resolve()
    if path.resolve() == allowed or not path.resolve().is_relative_to(allowed):
        raise ValueError('Dedicated private expression directory required')
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 1024):
            raise ValueError('Linked expression path forbidden')
    return path


def parse_request(value):
    if type(value) is not dict or set(value) != {'instruction', 'music_id', 'goals'}:
        raise ValueError('Exact instruction/music/goals request required')
    text = value['instruction']
    if type(text) is not str or not 3 <= len(text.strip()) <= 3000:
        raise ValueError('Instruction must contain3..3000 characters')
    text.encode('utf8')
    if value['music_id'] != 'phrase_original_pulse_v1' or type(value['music_id']) is not str:
        raise ValueError('Select the owned eight-second music')
    return {'instruction': text.strip(), 'music_id': value['music_id'],
            'goals': parse_goals(value['goals'])}


def unit_expression():
    return parse_expression({'schema': 'pulsemorph.phrase_expression.v1',
        'order': ['wave', 'high5'], 'heading_policy': 'align_join_heading',
        **{field: 1. for field in GAIN_FIELDS},
        'rationale': 'Fresh same-profile unit baseline; physical component goals remain separate.'})


def source_identity(runtime, music):
    return {'schema': 'pulsemorph.expression_source_identity.v1',
            'implementation_sha256': fingerprint(runtime._implementation),
            'environment_sha256': fingerprint(runtime.environment),
            'music_sha256': music['sha256'], 'source_id': 'phrase_wave_high5_v1',
            'duration_s': 8, 'sample_period_s': .02}


def public_trial(trial):
    if trial is None:
        return None
    keys = ('id', 'index', 'status', 'expression', 'metrics', 'diagnostics', 'evaluation', 'reason')
    return {k: copy.deepcopy(trial[k]) for k in keys if k in trial}


def public_job(job):
    keys = ('id', 'status', 'created_at', 'updated_at', 'instruction', 'music_id', 'goals',
            'events', 'plans', 'selected_id', 'result', 'video_url', 'export_url', 'error',
            'integration_status', 'new_model_calls', 'model_call_attempts', 'model_completion_unconfirmed',
            'new_physical_trials', 'worker_attempts', 'physical_completion_unconfirmed',
            'measurement_counts_unconfirmed', 'finalized')
    result = {k: copy.deepcopy(job.get(k)) for k in keys}
    result['baseline'] = public_trial(job.get('baseline'))
    result['trials'] = [public_trial(t) for t in job.get('trials', [])]
    return result


class ExpressionManager:
    def __init__(self, root, runtime, provider, music, *, jobs_dir=JOB_ROOT,
                 renderer=render_comparison, integrations=None):
        self.root, self.runtime, self.provider, self.music = Path(root).resolve(), runtime, provider, music
        self.jobs_dir = safe_directory(jobs_dir, CACHE)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer
        self.integrations = integrations or AppIntegrations(self.root,
            self.root / 'artifacts/runs/agentic/expression_studio_v1')
        self._lock, self._pool = threading.RLock(), ThreadPoolExecutor(max_workers=1)
        self._jobs, self._cancel, self._timers, self._deadlines = {}, {}, {}, {}
        self._live_provider_observed = False
        self.pins = self.implementation_pins()
        self.history_warnings = []
        self._load_history()

    def implementation_pins(self):
        paths = list(Path(__file__).parent.rglob('*.py'))
        paths += [p for p in Path(__file__).with_name('web').glob('*') if p.suffix in {'.js', '.css', '.html'}]
        paths += [self.root / 'src/pulsemorph/agentic/app/providers.py',
                  self.root / 'src/pulsemorph/agentic/app/integrations.py',
                  self.root / 'src/pulsemorph/agentic/app/phrase/music.py',
                  self.root / 'experiments/agentic/app/meterless_bridge.ts']
        return {**self.runtime._implementation, **{str(p.resolve()): sha(p) for p in paths}}

    def _load_history(self):
        for path in sorted(self.jobs_dir.glob('*/state.json'))[-100:]:
            try:
                safe_directory(path, self.jobs_dir)
                if path.stat().st_size > 2 * 1024**2 or path.is_symlink():
                    raise ValueError('Bounded ordinary job state required')
                job = json.loads(path.read_bytes())
                request = json.loads(path.with_name('request.json').read_bytes())
                if parse_request(request) != {k: job[k] for k in request} or job['id'] != path.parent.name:
                    raise ValueError('Stored request assignment differs')
                if (job.get('schema') != 'pulsemorph.expression_job.v1' or
                        type(job.get('created_at')) is not str or type(job.get('updated_at')) is not str or
                        type(job.get('events')) is not list or len(job['events']) > 1000 or
                        any(type(e) is not dict or type(e.get('message')) is not str for e in job['events']) or
                        type(job.get('plans')) is not list or len(job['plans']) > MAX_MODELS or
                        type(job.get('trials')) is not list or len(job['trials']) > MAX_CANDIDATES or
                        any(type(t) is not dict for t in job['trials']) or
                        job.get('baseline') is not None and type(job['baseline']) is not dict):
                    raise ValueError('Stored history shape differs')
                encoded(job)  # Reject nonfinite stored JSON as a whole.
                if job['status'] not in TERMINAL:
                    job.update(status='failed', error='PROCESS_INTERRUPTED', updated_at=now(), finalized=True)
                elif job.get('finalized') is False:
                    job.update(finalized=True, completion_note='Final memory commit interrupted on process restart.')
                self._jobs[job['id']] = job
            except Exception:
                self.history_warnings.append('invalid_or_partial_expression_history_preserved')

    def active_id(self):
        with self._lock:
            return next((j['id'] for j in self._jobs.values()
                         if j['status'] not in TERMINAL or j.get('finalized') is False), None)

    def status(self):
        runtime, provider = self.runtime.status(), self.provider.status()
        provider = {**provider, 'authentication_verified': self._live_provider_observed,
                    'authentication_basis': 'completed_live_call_this_session' if self._live_provider_observed else 'not_observed'}
        current = self.implementation_pins() == self.pins
        return {'ready': current and runtime['available'] and provider['enabled'], 'runtime': runtime,
                'provider': provider, 'active_job': self.active_id(),
                'history_warnings': list(self.history_warnings),
                'integrations': self.integrations.status(),
                'music': [{**{k: v for k, v in m.items() if k != 'path'},
                           'url': '/media/music/' + m['id'] + '.wav'}
                          for m in self.music.catalog() if m['id'] == 'phrase_original_pulse_v1'],
                'goal_scope': {'tolerance': .08, 'bounds': [.75, 1.25],
                    'labels': ['Ortalama eklem farkı', 'Zamana bağlı eklem değişimi'],
                    'window_s': [5.2, 7.2], 'sample_count': 101, 'arm': 'High5 sol kol'},
                'budgets': {'model_calls': 3, 'candidate_proposals': 3, 'physical_trials_including_baseline': 4,
                            'render_only_FK': 400, 'physics_steps_max': 32000}}

    def get(self, identity, *, private=False):
        with self._lock:
            job = copy.deepcopy(self._jobs[identity])
        return job if private else public_job(job)

    def list_jobs(self):
        with self._lock:
            return [public_job(j) for j in sorted(self._jobs.values(),
                    key=lambda j: j['created_at'], reverse=True)[:100]]

    def _update(self, identity, **changes):
        with self._lock:
            job = self._jobs[identity]
            job.update(changes, updated_at=now())
            path = self.jobs_dir / identity / 'state.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(encoded(job))
            temporary.replace(path)

    def _event(self, identity, kind, message):
        with self._lock:
            events = copy.deepcopy(self._jobs[identity]['events'])
            events.append({'seq': len(events), 'at': now(), 'kind': kind, 'message': message})
            self._update(identity, events=events)

    def _check(self, identity):
        if self._cancel[identity].is_set():
            raise RuntimeError('JOB_CANCELLED')
        if time.monotonic() >= self._deadlines[identity]:
            self._cancel[identity].set()
            raise RuntimeError('JOB_WALL_LIMIT')
        if self.implementation_pins() != self.pins:
            raise RuntimeError('SOURCE_CHANGED')
        self.runtime._unchanged()

    def create(self, value):
        request = parse_request(value)
        with self._lock:
            if self.active_id():
                raise ValueError('Another expression job is active')
            if not self.runtime.status()['available'] or not self.provider.status()['enabled']:
                raise ValueError('Provider or physical profile unavailable')
            identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_') + uuid.uuid4().hex[:8]
            directory = safe_directory(self.jobs_dir / identity, self.jobs_dir)
            directory.mkdir(exist_ok=False)
            write_new(directory / 'request.json', request)
            job = {'schema': 'pulsemorph.expression_job.v1', 'id': identity, **request,
                   'created_at': now(), 'updated_at': now(), 'status': 'queued', 'events': [],
                   'plans': [], 'baseline': None, 'trials': [], 'selected_id': None, 'result': None,
                   'video_url': None, 'export_url': None, 'error': None, 'source_pins': self.pins,
                   'new_model_calls': 0, 'model_call_attempts': 0, 'new_physical_trials': 0,
                   'finalized': False,
                   'integration_status': self.integrations.status()}
            self._jobs[identity], self._cancel[identity] = job, threading.Event()
            self._deadlines[identity] = time.monotonic() + WALL_SECONDS
            timer = threading.Timer(WALL_SECONDS, self._cancel[identity].set)
            timer.daemon = True
            self._timers[identity] = timer
            timer.start()
            self._update(identity)
            self._pool.submit(self._run, identity)
        return identity

    def cancel(self, identity):
        with self._lock:
            if identity not in self._jobs:
                raise KeyError('Unknown expression job')
            if identity in self._cancel and self._jobs[identity]['status'] not in TERMINAL:
                self._cancel[identity].set()
                self._event(identity, 'cancel_requested', 'İptal istendi; yürüyen işlem durduruluyor.')

    def _remember(self, identity, trial=None):
        try:
            job = self.get(identity, private=True)
            if trial is None:
                self.integrations.record_job({'id': identity, 'source_id': SOURCE_ID,
                    'status': job['status'], 'goals': job['goals'], 'selected_id': job['selected_id'],
                    'result': job['result'], 'instruction': job['instruction']})
            else:
                evaluation = trial.get('evaluation') or {}
                self.integrations.record_trial(identity, {'id': trial['id'], 'index': trial['index'],
                    'source_id': SOURCE_ID, 'expression': trial['expression'], 'goals': job['goals'],
                    'status': trial['status'], 'evaluation': evaluation,
                    'source_identity': trial.get('source_identity'),
                    'metrics': {**(trial.get('metrics') or {}),
                        'integrity_passed': trial.get('integrity_passed', False),
                        'intent_passed': evaluation.get('intent_passed', False)}})
        except Exception:
            self._event(identity, 'memory_unavailable', 'Bellek kaydı kullanılamadı; hareket kanıtı yerelde korunuyor.')
        self._update(identity, integration_status=self.integrations.status())

    def _context(self, identity, music):
        if sha(PILOT) != PILOT_SHA:
            raise RuntimeError('PINNED_PILOT_CHANGED')
        pilot = json.loads(PILOT.read_bytes())['trials']
        denominator = pilot[0]['diagnostics'][2]
        observations = [{'expression': t['expression'],
                'qualification': 'Historical independently audited expression_motor_v1; one trajectory per control, not calibration or an attainable region. Not a fresh baseline.',
                'source_completed_sha256': PILOT_SHA,
                'mean_pose_ratio': t['diagnostics'][2]['mean_pose_offset_rms_rad'] / denominator['mean_pose_offset_rms_rad'],
                'temporal_ratio': t['diagnostics'][2]['centered_temporal_rms_rad'] / denominator['centered_temporal_rms_rad']}
                for t in pilot]
        try:
            memory = self.integrations.query_sources(self.get(identity)['instruction'], 3,
                                                    source_id=SOURCE_ID, body_id='g1')
        except Exception:
            memory = {'memory': [], 'state': 'unavailable'}
        remembered = []
        for row in memory.get('memory', []):
            evaluation = row.get('evaluation') or {}
            ratios = evaluation.get('actual_ratios') or {}
            if row.get('source_identity') == source_identity(self.runtime, music) and evaluation.get('assessable') is True:
                remembered.append({'expression': row['expression'],
                    'mean_pose_ratio': ratios['mean_pose_ratio'], 'temporal_ratio': ratios['temporal_ratio'],
                    'qualification': 'Retrieved Meterless same-profile finite saved measurement; context only, not a calibrated feasible region or new baseline.',
                    'source_job_id': row.get('job_id'), 'source_trial_id': row['id']})
        observations = (remembered + observations)[:3]
        summary_keys = ('schema', 'sha256', 'duration_s', 'sample_rate', 'sample_frames', 'channels',
                        'pcm_bits', 'rms', 'peak_amplitude', 'accent_time_s', 'accent_measurement',
                        'onset_times_s', 'sections', 'bpm', 'bpm_source', 'limitations')
        return {'music_summary': {k: music[k] for k in summary_keys if k in music},
                'source_descriptor': describe_bases(self.runtime.assets.references()),
                'observations': observations,
                'constraints': {'adjustable_controls': ['second_left_pose_gain', 'second_left_motion_gain'],
                    'frozen_other_gains': 1., 'heading': 'align_join_heading', 'execution_s': 8,
                    'goal_metric_window_s': [5.2, 7.2], 'sample_count': 101,
                    'model_call_limit': MAX_MODELS, 'candidate_proposal_limit': MAX_CANDIDATES,
                    'retrieval': {'memory_records_retrieved': len(memory.get('memory', [])),
                        'memory_observations_used': len(remembered[:3]),
                        'zetaris': memory.get('zetaris', {'executed': False, 'state': 'unavailable'})},
                    'bounds_are_reference_only': True}}

    def _physical(self, identity, expression, index, music):
        self._check(identity)
        label = 'baseline' if index == -1 else 'candidate_%02d' % index
        attempts = self.jobs_dir / identity / 'physical_attempts'
        attempts.mkdir(exist_ok=True)
        write_new(attempts / (label + '.json'), {'index': index, 'expression': expression,
            'compile_slot_consumed': True, 'worker_completion_not_yet_known': True})
        trial = self.runtime.run_trial(expression, self.jobs_dir / identity / 'trials' / label,
            identity, index, {'available': True, 'path': music['path'], 'sha256': music['sha256'],
                'duration_s': 8, 'sample_rate': 48000, 'observed_through_sample': 384000}, self._cancel[identity])
        self.runtime._verify_trial(trial)
        trial.update(status='measured', source_identity=source_identity(self.runtime, music))
        self._sync_counts(identity)
        return trial

    def _sync_counts(self, identity):
        directory = self.jobs_dir / identity
        calls = list((directory / 'model').glob('*/attempt.json'))
        receipts = []
        for path in (directory / 'model').glob('*/receipt.json'):
            record = json.loads(path.read_bytes())
            receipts.append(record.get('live'))
        started = len(list((directory / 'trials').glob('*/worker_started.json')))
        results = [json.loads(p.read_bytes()) for p in (directory / 'trials').glob('*/result.json')]
        complete = sum(r.get('status') == 'completed' and r.get('integrity_passed') is True for r in results)
        self._update(identity, new_model_calls=sum(v is True for v in receipts),
            model_call_attempts=len(calls), model_completion_unconfirmed=len(calls) - len(receipts),
            new_physical_trials=complete, worker_attempts=started,
            physical_completion_unconfirmed=max(0, started - complete),
            reference_compile_attempts=len(list((directory / 'physical_attempts').glob('*.json'))))

    def _run(self, identity):
        try:
            self._check(identity)
            job = self.get(identity, private=True)
            music = self.music.get(job['music_id'])
            context = self._context(identity, music)
            write_new(self.jobs_dir / identity / 'context.json', context)
            self._remember(identity)
            feedback = None
            for index in range(MAX_MODELS):
                self._check(identity)
                self._update(identity, status='planning' if index == 0 else 'feedback')
                self._event(identity, 'model', 'Müzik ve talimat planlanıyor.' if index == 0 else 'Model gerçek fizik geri bildirimini değerlendiriyor.')
                self._update(identity, model_call_attempts=index + 1)
                proposal = plan_expression(self.provider,
                    {'instruction': job['instruction'], 'goals': job['goals']}, context,
                    self.jobs_dir / identity / 'model' / ('call_%02d' % index),
                    cancel_event=self._cancel[identity], feedback=feedback)
                self._check(identity)
                if proposal['receipt'].get('live') is not True:
                    raise RuntimeError('LIVE_PROVIDER_RECEIPT_REQUIRED')
                self._live_provider_observed = True
                self._sync_counts(identity)
                plan = {k: proposal[k] for k in ('supported', 'intent_summary', 'unsupported_requests',
                                                'expression', 'goals', 'receipt')}
                plans = self.get(identity)['plans'] + [{'index': index, **plan}]
                self._update(identity, plans=plans)
                if not proposal['locally_supported']:
                    self._update(identity, status='unsupported', result='unsupported')
                    self._event(identity, 'unsupported', 'İstek bu deneyin iki High5 sol kol özelliğinin dışında; fizik başlatılmadı veya yeni aday durduruldu.')
                    return
                if index == 0:
                    self._update(identity, status='baseline')
                    self._event(identity, 'baseline', 'Değiştirilmemiş başlangıç aynı profille fizikte ölçülüyor.')
                    baseline = self._physical(identity, unit_expression(), -1, music)
                    bridge_result = bridge(baseline['paths']['directory'], HISTORICAL_UNIT)
                    write_new(self.jobs_dir / identity / 'unit_bridge.json', bridge_result)
                    baseline['evaluation'] = evaluate_goals(baseline, baseline, job['goals'])
                    self._update(identity, baseline=baseline)
                    self._remember(identity, baseline)
                    if not baseline['evaluation']['physical_gate_passed'] or not baseline['evaluation']['assessable']:
                        raise RuntimeError('BASELINE_UNASSESSABLE_OR_PHYSICAL_GATE')
                baseline = self.get(identity, private=True)['baseline']
                self._update(identity, status='trial')
                self._event(identity, 'candidate', 'Aday %d derlenip fizikte ölçülüyor.' % (index + 1))
                try:
                    trial = self._physical(identity, proposal['expression'], index, music)
                    trial['evaluation'] = evaluate_goals(trial, baseline, job['goals'])
                except ProgramCompileError as error:
                    trial = {'id': identity + '__trial_' + str(index), 'index': index,
                        'expression': proposal['expression'], 'status': 'rejected', 'metrics': None,
                        'diagnostics': None, 'evaluation': None, 'reason': 'reference_compile_rejected',
                        'rejection_report': error.report}
                    self._event(identity, 'compile_rejected', 'Aday sabit referans sınırlarını geçemedi; fizik çalıştırılmadı.')
                trials = self.get(identity, private=True)['trials'] + [trial]
                selected = select_final(trials, baseline, job['goals'])
                self._update(identity, trials=trials, selected_id=selected['trial']['id'])
                self._remember(identity, trial)
                if trial.get('evaluation') is not None and trial['evaluation']['intent_passed']:
                    break
                feedback = {'schema': 'pulsemorph.expression_feedback.v1',
                    'fixed_goals': job['goals'], 'baseline': public_trial(baseline),
                    'trials': [{**public_trial(t), **({'rejection_report': t['rejection_report']} if t['status'] == 'rejected' else {})} for t in trials],
                    'candidate_budget_remaining': MAX_CANDIDATES - index - 1,
                    'qualification': 'Original fixed targets/tolerance remain unchanged. Rejections are not physical observations. All completed physical measurements and cross-arm diagnostics are included.'}
            self._check(identity)
            final = self.get(identity, private=True)
            selected = select_final(final['trials'], final['baseline'], job['goals'])
            result = 'success' if selected['evaluation']['intent_passed'] else 'partial'
            self._update(identity, status='rendering', selected_id=selected['trial']['id'], result=result)
            self._event(identity, 'render', 'Seçilen sonuç ve başlangıç kayıtlı fizik durumlarından görüntüleniyor.')
            media = self.renderer(self.root, self.runtime, final['baseline'], selected['trial'], music,
                                  self.jobs_dir / identity / 'media', cancel_event=self._cancel[identity])
            self._check(identity)
            self._export(identity, selected['trial'])
            manifest = json.loads(Path(media['manifest']).read_bytes())
            if (manifest['baseline_id'] != final['baseline']['id'] or
                    manifest['selected_id'] != selected['trial']['id'] or
                    manifest['comparison_sha256'] != sha(media['comparison'])):
                raise RuntimeError('MEDIA_ASSIGNMENT_CHANGED')
            self._update(identity, status='completed', video_url='/media/jobs/' + identity + '/comparison.mp4',
                         export_url='/api/jobs/' + identity + '/export',
                         media_integrity={'video_sha256': sha(media['comparison']),
                             'manifest_sha256': sha(media['manifest']),
                             'export_sha256': sha(self.jobs_dir / identity / 'reference.zip')})
            self._event(identity, 'complete', 'İki fizik hedefi ölçüm toleransında karşılandı.' if result == 'success' else 'Deneme bütçesi tamamlandı; en iyi ölçülen sonuç kısmi olarak seçildi.')
        except Exception:
            expired = time.monotonic() >= self._deadlines[identity]
            cancelled = self._cancel[identity].is_set() and not expired
            self._update(identity, status='cancelled' if cancelled else 'failed',
                         error=None if cancelled else ('JOB_WALL_LIMIT' if expired else 'EXPRESSION_JOB_STOPPED'))
            self._event(identity, 'cancelled' if cancelled else 'failed',
                        'İptal edildi; kısmi kayıtlar korundu.' if cancelled else 'İş durduruldu; hedefler değiştirilmeden kısmi kayıtlar korundu.')
        finally:
            self._timers[identity].cancel()
            try:
                self._sync_counts(identity)
            except Exception:
                self._update(identity, measurement_counts_unconfirmed=True)
            self._remember(identity)
            self._update(identity, finalized=True)

    def _export(self, identity, selected):
        self.runtime._verify_trial(selected)
        directory = self.jobs_dir / identity
        summary = public_job(self.get(identity, private=True))
        summary.update(status='completed', video_url='/media/jobs/' + identity + '/comparison.mp4',
                       export_url='/api/jobs/' + identity + '/export')
        summary['export_scope'] = 'Private selected common reference and measured job summary, retaining original local provenance paths; not a portable/public package or rights grant. No model weights/body meshes.'
        with zipfile.ZipFile(directory / 'reference.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('summary.json', encoded(summary))
            for name in ('manifest.json', 'arrays.npz'):
                archive.write(Path(selected['paths']['common_reference']) / name, 'common_reference/' + name)
        with zipfile.ZipFile(directory / 'reference.zip') as archive:
            if archive.testzip() is not None:
                raise RuntimeError('EXPORT_CRC_FAILED')

    def media_path(self, identity, kind):
        if re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identity) is None:
            raise KeyError('Unknown expression job')
        job = self.get(identity, private=True)
        if job['status'] != 'completed':
            raise ValueError('Only completed expression media is available')
        # Replay serves the immutable render accepted at job completion. This is
        # a current profile/media-byte guard, not a new full raw-trial audit.
        self.runtime._unchanged()
        if any(job['source_pins'].get(p) != h for p, h in self.runtime._implementation.items()):
            raise ValueError('Recorded physics profile differs from current profile')
        integrity = job['media_integrity']
        media = self.jobs_dir / identity / 'media'
        if sha(media / 'manifest.json') != integrity['manifest_sha256']:
            raise ValueError('Recorded media manifest changed')
        name = 'media/comparison.mp4' if kind == 'video' else 'reference.zip'
        path = safe_directory(self.jobs_dir / identity / name, self.jobs_dir)
        expected = integrity['video_sha256' if kind == 'video' else 'export_sha256']
        if sha(path) != expected:
            raise ValueError('Recorded media/export bytes changed')
        return path
