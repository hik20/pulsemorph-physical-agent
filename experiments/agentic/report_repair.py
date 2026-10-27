"""Read-only posthoc repair report; requires completed, independently audited physics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

TASKS = ('wave_a', 'high5_b')
METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
LABELS = {'incumbent':'Old incumbent', 'surrogate_search':'Numeric minimax',
          'llm_feedback':'LLM + feedback', 'llm_feedback_off':'LLM feedback off'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def linked(record):
    path = Path(record['path'])
    if sha(path) != record['sha256']:
        raise ValueError('evidence hash mismatch: '+str(path))
    return read(path)


def aggregate(rows):
    physical = all(r['physical_gate_passed'] for r in rows.values())
    utilities = [r['primary']['utility'] for r in rows.values()]
    worst = max(utilities) if all(u is not None for u in utilities) else None
    squares = [r['primary']['utility_l2'] for r in rows.values()]
    return dict(physical_gate_passed=physical, worst_normalized_error=worst,
                full_margin=None if worst is None else 1-worst,
                squared_error=sum(squares) if all(s is not None for s in squares) else None)


def rank(candidate):
    a = candidate['aggregate']
    return (not a['physical_gate_passed'],
            math.inf if a['worst_normalized_error'] is None else a['worst_normalized_error'],
            math.inf if a['squared_error'] is None else a['squared_error'], candidate['candidate_id'])


def peak_ranking_diagnostic(evaluation, trial):
    """Posthoc top-two detected accent events; never a new selection metric."""
    import numpy as np
    path=Path(trial)/'state_times.npy'
    manifest=read(Path(trial)/'manifest.json')
    if sha(path) != manifest['artifact_hashes']['state_times.npy']:
        raise ValueError('diagnostic state clock hash mismatch')
    times=np.load(path,allow_pickle=False)
    accent=evaluation['accent']
    smooth=np.asarray(accent['smoothed_rms_speed_rad_s'])
    if times.shape != smooth.shape:
        raise ValueError('accent diagnostic clock shape mismatch')
    events=[]
    for time in accent['event_times_s']:
        if 1.8 <= time <= 3.2:
            index=int(np.argmin(np.abs(times-time)))
            if abs(float(times[index])-time)>1e-9:
                raise ValueError('accent event is not on saved state clock')
            events.append(dict(time_s=float(times[index]),height_rad_s=float(smooth[index])))
    events.sort(key=lambda item:(-item['height_rad_s'],item['time_s']))
    return dict(diagnostic_only=True,window_s=[1.8,3.2],state_times=pin(path),
                detected_window_events=len(events),top_two=events[:2],
                top_two_height_gap_rad_s=None if len(events)<2 else events[0]['height_rad_s']-events[1]['height_rad_s'],
                unavailable_reason='fewer than two detected events in window' if len(events)<2 else None,
                interpretation='Posthoc peak ranking fragility; not tolerance, probability, acceptance or selection metric.')


def collect(panel, audit):
    panel = Path(panel).resolve()
    protocol, summary, accepted = read(panel/'protocol.json'), read(panel/'summary.json'), read(audit)
    digest = sha(panel/'protocol.json')
    addendum_path=panel.parent/'repair_legacy_fields_review_v1/report.json'
    annotations_path=addendum_path.with_name('annotations.json')
    disclosure=read(addendum_path)
    annotations=read(annotations_path)
    if (disclosure.get('status')!='provenance_incomplete_disclosure'
            or disclosure.get('bounded_impact_checks_passed') is not True
            or disclosure.get('raw_metadata_semantically_correct') is not False
            or disclosure.get('inputs_unchanged_during_review') is not True
            or disclosure.get('errors')!=[] or len(annotations)!=250
            or disclosure['protocol']!=pin(panel/'protocol.json')
            or disclosure['selection']!=pin(panel/'selection.json')):
        raise ValueError('protocol-bound bounded legacy metadata exception disclosure required')
    if (not summary.get('completed') or summary.get('errors') != []
            or summary.get('protocol_sha256') != digest
            or accepted.get('audit_passed') is not True
            or accepted.get('input_protocol_sha256') != digest
            or Path(accepted['input_panel']).resolve() != panel
            or accepted.get('panel_evidence_unchanged') is not True):
        raise ValueError('completed panel and passed independent audit bound to this protocol required')
    conditions = [s['scenario_id'] for s in protocol['scenarios']]
    dev = protocol['development_condition_ids']
    if len(conditions) != 13 or len(dev) != 4 or len(summary['winners']) != 6:
        raise ValueError('exact two-task, three-method, thirteen-condition panel required')
    base = Path(protocol['baseline_panel'])
    rows, comparisons, curves, candidate_batches = [], [], [], []
    for task in TASKS:
        incumbent = {s: read(base/'evaluations'/(s+'__selected_'+task+'.json')) for s in conditions}
        outputs = [('incumbent', 'incumbent', incumbent, None)]
        for method in METHODS:
            winner = next(w for w in summary['winners'] if (w['task_id'],w['method']) == (task,method))
            results = {s:linked(r['evaluation']) for s,r in winner['condition_records'].items()}
            if set(results) != set(conditions):
                raise ValueError('full condition inventory mismatch')
            outputs.append((method,winner['selected_candidate_id'],results,winner))
            best = dict(candidate_id='incumbent', aggregate=aggregate({s:incumbent[s] for s in dev}))
            curves.append(dict(task_id=task,method=method,candidate_calls=0,**best))
            for index in range(8):
                identity = f'{task}__{method}__{index}'
                candidate = read(panel/'candidates'/(identity+'.json'))
                measured = {s:linked(r['evaluation']) for s,r in candidate['condition_records'].items()}
                item = dict(candidate_id=identity, aggregate=aggregate(measured))
                candidate_batches.append(dict(task_id=task,method=method,index=index,
                    candidate=pin(panel/'candidates'/(identity+'.json')),
                    condition_records=candidate['condition_records'],
                    failed_condition_ids=[s for s,r in measured.items() if not r['primary']['joint_success']],
                    condition_outcomes={s:dict(physical_gate_passed=r['physical_gate_passed'],
                        primary=r['primary'],quality_diagnostics=r['quality_diagnostics']) for s,r in measured.items()},
                    **item))
                best = min((best,item), key=rank)
                curves.append(dict(task_id=task,method=method,candidate_calls=index+1,**best))
        for method,identity,results,winner in outputs:
            local = []
            for condition in conditions:
                r = results[condition]
                if r['integrity_passed'] is not True:
                    raise ValueError('integrity failure cannot be plotted')
                score = r['primary']
                record=None if winner is None else winner['condition_records'][condition]
                trial=(base/'trials'/(condition+'__selected_'+task) if record is None or record.get('cached_prior')
                       else panel/'trials'/record['trial_id'])
                row = dict(task_id=task,method=method,candidate_id=identity,condition_id=condition,
                    development_condition=condition in dev, physical_gate_passed=r['physical_gate_passed'],
                    primary_passed=score['joint_success'],worst_normalized_error=score['utility'],
                    normalized_margin=None if score['utility'] is None else 1-score['utility'],
                    accent_goal_time_margin_s=None if score['feature_errors'][-1] is None else
                        score['feature_tolerances'][-1]-abs(score['feature_errors'][-1]),
                    features=score['features'],feature_errors=score['feature_errors'],
                    feature_tolerances=score['feature_tolerances'],quality_diagnostics=r['quality_diagnostics'],
                    peak_ranking_diagnostic=peak_ranking_diagnostic(r,trial))
                rows.append(row);local.append(row)
            gaps=[(r['peak_ranking_diagnostic']['top_two_height_gap_rad_s'],r['condition_id']) for r in local
                  if r['peak_ranking_diagnostic']['top_two_height_gap_rad_s'] is not None]
            minimum_gap=min(gaps) if gaps else None
            time_margins=[(r['accent_goal_time_margin_s'],r['condition_id']) for r in local if r['accent_goal_time_margin_s'] is not None]
            minimum_time_margin=min(time_margins) if time_margins else None
            comparisons.append(dict(task_id=task,method=method,candidate_id=identity,
                control_tool=None if winner is None else winner['control_tool'],
                parameters=None if winner is None else winner['parameters'],
                nominal_passed=next(r['primary_passed'] for r in local if r['condition_id']=='nominal'),
                nonzero_passed=sum(r['primary_passed'] for r in local if r['condition_id']!='nominal'),
                full_passed=sum(r['primary_passed'] for r in local),
                minimum_top_two_peak_gap_rad_s=None if minimum_gap is None else minimum_gap[0],
                minimum_peak_gap_condition_id=None if minimum_gap is None else minimum_gap[1],
                minimum_accent_goal_time_margin_s=None if minimum_time_margin is None else minimum_time_margin[0],
                minimum_accent_goal_time_margin_condition_id=None if minimum_time_margin is None else minimum_time_margin[1],
                peak_gap_unavailable_conditions=[r['condition_id'] for r in local if r['peak_ranking_diagnostic']['unavailable_reason']],
                failed_condition_ids=[r['condition_id'] for r in local if not r['primary_passed']],
                **aggregate(results)))
    return dict(schema='pulsemorph.agentic_repair_posthoc.v1',protocol=pin(panel/'protocol.json'),
        reporter_source=pin(__file__),
        model_provenance=dict(requested_model='gpt-6.1-sol',provider_revision=None,
            provider_tokens=None,provider_billed_cost=None,reasoning_effort_override=None,
            interpretation='Requested alias recorded in collaboration calls; immutable provider weight revision unavailable. Same unspecified effort settings in both arms. Motor ONNX/checkpoint hashes are pinned separately.'),
        legacy_metadata_disclosure=pin(addendum_path),legacy_metadata_annotations=pin(annotations_path),
        raw_metadata_semantically_correct=False,
        raw_execution_metadata_complete=False,legacy_metadata_disclosure_required=True,
        acceptance_scope='numerical_active_lineage_with_legacy_metadata_exception',
        metadata_exception='Raw trial proposals inherited seed trial_index, brief_reason, created_at_utc and search_decision. Separate all250 review bounded current identity/tool/parameter and physics-input impact; raw metadata is not semantically correct. Current proposals/responses are authoritative; no raw rewriting, rescore or rerank.',
        independent_audit=pin(audit),summary=pin(panel/'summary.json'),conditions=conditions,
        comparisons=comparisons,best_so_far=curves,candidate_batches=candidate_batches,condition_rows=rows,
        scope='Known warm-start development: two selected tasks, two sources, all thirteen conditions known; no held-out/generalization claim.',
        equal_prior='Every method: 24 old nominal candidates + 2 nominal identity tool records + incumbent 13 conditions. Incumbent nominal repeats a prior trace, not independent evidence.',
        numeric_method='Adaptive mixed-tool weighted affine minimax, residual-based trust region; surrogate transfer assumptions are not physics evidence.',
        budget=dict(new_model_proposal_turns=32,engineering_agent_turns_excluded=True,new_physics=250,
                    runtime_gate=4,candidate=192,remaining_confirmation=54,control_steps=50000,physics_steps=1000000),
        reuse='Full13 reuses four development rows; selected incumbent reuses four historical prior rows. Remaining nine are fresh known-development confirmation; no reselection.',
        diagnostic_scope='Applied PD target steps and joint acceleration finite differences sampled at 20ms are separate diagnostics; omit within-step acceleration. No human quality, recovery, motor learning or new-body claim.')


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for task,ax in zip(TASKS,axes):
        for method in METHODS:
            points=[r for r in data['best_so_far'] if (r['task_id'],r['method'])==(task,method)]
            ax.plot([r['candidate_calls'] for r in points],
                    [np.nan if r['aggregate']['worst_normalized_error'] is None else r['aggregate']['worst_normalized_error'] for r in points],marker='o',label=LABELS[method])
            for r in points:
                if not r['aggregate']['physical_gate_passed']:
                    ax.scatter(r['candidate_calls'],r['aggregate']['worst_normalized_error'],marker='x',s=100,color='red')
        ax.axhline(1,color='black',ls='--',lw=1);ax.set(title=task,xlabel='New candidate calls (0 = old incumbent; 4 physics/call)',ylabel='Four-condition worst normalized error',xticks=range(9))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Known development: physical-first best-so-far (red × = physical failure)')
    fig.savefig(output/'development_best_so_far.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,8),constrained_layout=True)
    methods=('incumbent',*METHODS)
    for task,ax in zip(TASKS,axes):
        selected={(r['condition_id'],r['method']):r for r in data['condition_rows'] if r['task_id']==task}
        matrix=np.array([[np.nan if selected[s,m]['worst_normalized_error'] is None else selected[s,m]['worst_normalized_error'] for m in methods] for s in data['conditions']])
        im=ax.imshow(matrix,cmap='RdYlGn_r',vmin=0,vmax=2,aspect='auto')
        for i,s in enumerate(data['conditions']):
            for j,m in enumerate(methods):
                r=selected[s,m];value=matrix[i,j]
                label='missing' if np.isnan(value) else f'{value:.2f}'
                label += '\nPASS' if r['primary_passed'] else '\nFAIL'
                if not r['physical_gate_passed']:label+=' phys'
                ax.text(j,i,label,ha='center',va='center',fontsize=8)
        ax.set(title=task,xticks=range(4),xticklabels=[LABELS[m] for m in methods],yticks=range(13),yticklabels=data['conditions'])
        ax.tick_params(axis='x',rotation=25)
    fig.colorbar(im,ax=axes,label='Worst normalized error (acceptance ≤ 1; colors clipped at 2)',shrink=.65)
    fig.suptitle('Frozen outputs, all 13 known development conditions; all failures retained')
    fig.savefig(output/'full13_heatmaps.png',dpi=180);plt.close(fig)


def report(panel,audit,output):
    data=collect(panel,audit)
    output=Path(output).resolve()
    panel=Path(panel).resolve()
    if output.exists() or output==panel or panel in output.parents:
        raise ValueError('new separate report directory required')
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__),output/'report_repair.py')
    (output/'report.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
    render(data,output)
    lines=['# Repair development results','',data['scope'],'',data['metadata_exception'],
           'Acceptance scope: numerical_active_lineage_with_legacy_metadata_exception; raw_metadata_semantically_correct=false.',
           '',data['equal_prior'],'',data['numeric_method'],'',
           data['model_provenance']['interpretation'],'',
           '| Task | Method | Nominal | Nonzero /12 | Full /13 | Worst error | Margin 1−worst | Minimum peak gap rad/s | Failed condition IDs |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in data['comparisons']:
        fmt=lambda v:'missing' if v is None else f'{v:.4f}'
        lines.append(f"| {r['task_id']} | {LABELS[r['method']]} | {int(r['nominal_passed'])} | {r['nonzero_passed']} | {r['full_passed']} | {fmt(r['worst_normalized_error'])} | {fmt(r['full_margin'])} | {fmt(r['minimum_top_two_peak_gap_rad_s'])} | {', '.join(r['failed_condition_ids']) or 'none'} |")
    lines += ['', '32 new model proposal turns (engineering calls excluded). 250 new physics = 4 runtime gate + 192 candidate + 54 remaining confirmation; 50,000 control / 1,000,000 physics steps. Tokens and billed cost unmeasured.', '',data['reuse'],'',data['diagnostic_scope'],'',
              'Physical-first ranking precedes error and squared-error ties. Curves include the old incumbent at every call; they can rise in error when a physically valid candidate replaces a physical failure. Quality diagnostics are preserved per condition in report.json and never enter selection.', '',
              'The selector minimizes worst error after the physical gate; it does not maximize passing-condition count. A lower worst error may therefore accompany fewer passing conditions. Full worst error, pass counts, and accent goal-time margins are distinct quantities; confirmation never triggers reselection.', '',
              'Posthoc peak ranking diagnostic: top two detected accent event heights within 1.8–3.2s use saved actual state times and smoothed RMS speeds. Fewer than two events yields NA with reason. Per-condition heights/gaps and minimum-gap condition IDs are in report.json. This diagnostic never changes selection. A higher pass count alone does not establish larger goal-time acceptance margin or reduced peak ranking fragility.', '',
              f"Independent audit: `{data['independent_audit']['path']}`; SHA256 `{data['independent_audit']['sha256']}`.",
              f"Protocol SHA256: `{data['protocol']['sha256']}`.", '', '![Development curves](development_best_so_far.png)', '', '![Full panel](full13_heatmaps.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report(args.panel,args.audit,args.output)
