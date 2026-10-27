"""Read-only posthoc figures for a completed finite initial-state panel.

This reporter never executes, repairs or rescores a physics trial. Independent
audit status is linked explicitly and is not inferred from panel completion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

TASKS = ('wave_a', 'wave_b', 'high5_a', 'high5_b')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def collect(panel, audit=None):
    import numpy as np
    panel = Path(panel).resolve()
    summary, protocol = read(panel/'summary.json'), read(panel/'protocol.json')
    if summary.get('panel_completed') is not True:
        raise ValueError('completed panel required; no figures generated for partial physics')
    if summary['protocol_sha256'] != sha(panel/'protocol.json'):
        raise ValueError('summary/protocol identity mismatch')
    scenarios = [s['scenario_id'] for s in protocol['scenarios']]
    if len(scenarios) != 13 or len(summary['rows']) != 52:
        raise ValueError('exact 13 by 4 completed outcome panel required')
    rows, baseline = [], {}
    feature_names = protocol['feature_names']
    for scenario in scenarios:
        for task in TASKS:
            identity = scenario + '__selected_' + task
            source = next(t['source_id'] for t in protocol['tasks'] if t['task_id'] == task)
            path = panel/'evaluations'/(identity+'.json')
            evaluation = read(path)
            if not evaluation['integrity_passed']:
                raise ValueError('integrity-invalid evaluation: ' + identity)
            matched = [r for r in summary['rows'] if r['trial_id'] == identity]
            if len(matched) != 1:
                raise ValueError('missing/duplicate summary row: ' + identity)
            for key in ('primary', 'secondary', 'physical_gate_passed', 'min_root_height_m', 'mean_joint_rmse_rad'):
                if matched[0][key] != evaluation[key]:
                    raise ValueError('summary/evaluation differs: ' + identity + ':' + key)
            values = {}
            for estimand in ('primary', 'secondary'):
                score = evaluation[estimand]
                margins = []
                for name, error, tolerance in zip(feature_names, score['feature_errors'], protocol['feature_tolerances']):
                    margins.append(dict(feature=name, signed_error=error, tolerance=tolerance,
                        normalized_abs_error=None if error is None else abs(error)/tolerance,
                        raw_margin=None if error is None else tolerance-abs(error),
                        normalized_margin=None if error is None else 1-abs(error)/tolerance))
                values[estimand] = dict(worst_normalized_error=score['utility'],
                    intent_passed=score['intent_passed'], joint_success=score['joint_success'], margins=margins)
            rows.append(dict(trial_id=identity, scenario_id=scenario, task_id=task, source_id=source,
                evaluation=pinned(path), physical_gate_passed=evaluation['physical_gate_passed'],
                min_root_height_m=evaluation['min_root_height_m'], mean_joint_rmse_rad=evaluation['mean_joint_rmse_rad'],
                accent=evaluation['accent'],
                quality_diagnostics=evaluation['quality_diagnostics'],
                **values))
            drift = []
            for dimension in evaluation['dimensions']:
                denominator = dimension['nominal_baseline_rms_rad']
                drift.append(dict(feature=dimension['name'], nominal_rms_B0_rad=denominator,
                    condition_rms_about_nominal_rad=dimension['condition_baseline_rms_about_nominal_rad'],
                    condition_rms_about_condition_rad=dimension['condition_baseline_rms_rad'],
                    Bcondition_about_c0_over_B0=dimension['condition_baseline_rms_about_nominal_rad']/denominator,
                    Bp_over_B0=dimension['condition_baseline_rms_rad']/denominator,
                    condition_center_minus_nominal_rad=dimension['condition_center_minus_nominal_rad']))
            key = source + ':' + scenario
            if key in baseline and baseline[key]['dimensions'] != drift:
                raise ValueError('same-source baseline drift differs across tasks')
            baseline[key] = dict(source_id=source, scenario_id=scenario, dimensions=drift,
                nominal_physical=evaluation['nominal_baseline_physical'],
                condition_physical=evaluation['condition_baseline_physical'])
    tasks = []
    for task in TASKS:
        selected = [r for r in rows if r['task_id'] == task]
        counts = {}
        for name, subset in [('nominal', [r for r in selected if r['scenario_id']=='nominal']),
                             ('nonzero', [r for r in selected if r['scenario_id']!='nominal'])]:
            counts[name] = dict(cases=len(subset), physical_passed=sum(r['physical_gate_passed'] for r in subset),
                primary_passed=sum(r['primary']['joint_success'] for r in subset),
                secondary_passed=sum(r['secondary']['joint_success'] for r in subset))
        worst = {}
        for estimand in ('primary', 'secondary'):
            worst[estimand] = []
            for feature in feature_names:
                available = [(r, next(m for m in r[estimand]['margins'] if m['feature']==feature)) for r in selected]
                available = [(r,m) for r,m in available if m['normalized_margin'] is not None]
                if available:
                    row, margin = min(available, key=lambda pair:pair[1]['normalized_margin'])
                    worst[estimand].append(dict(scenario_id=row['scenario_id'], **margin))
                else:
                    worst[estimand].append(dict(feature=feature, unavailable=True))
        tasks.append(dict(task_id=task, counts=counts, worst_feature_margins=worst,
            range_and_worst_scope='All13 cases: nominal plus12 nonzero; counts split separately.',
            unavailable_accent_cases=[r['scenario_id'] for r in selected if r['accent'].get('peak_time_s') is None],
            quality_diagnostic_ranges={name:[min(r['quality_diagnostics'][name] for r in selected),
                max(r['quality_diagnostics'][name] for r in selected)] for name in
                ('applied_pd_target_step_max_abs_rad','joint_acceleration_finite_difference_max_abs_rad_s2')},
            physical_range=dict(min_root_height_m=[min(r['min_root_height_m'] for r in selected), max(r['min_root_height_m'] for r in selected)],
                mean_joint_rmse_rad=[min(r['mean_joint_rmse_rad'] for r in selected), max(r['mean_joint_rmse_rad'] for r in selected)])))
    audit_record = None
    if audit is not None:
        audit_data = read(audit)
        if audit_data.get('input_protocol_sha256') != sha(panel/'protocol.json'):
            raise ValueError('independent audit refers to a different protocol')
        audit_record = dict(**pinned(audit), audit_passed=audit_data.get('audit_passed'))
    contacts=[]
    for scenario in scenarios:
        for source in ('wave','high5'):
            manifest_path=panel/'trials'/(scenario+'__calibration_'+source)/'manifest.json'
            initial=read(manifest_path)['initial_perturbation']
            contacts.append(dict(scenario_id=scenario,source_id=source,
                initial_contact_count=initial['initial_contact_count'],
                initial_minimum_contact_distance_m=initial['initial_minimum_contact_distance_m']))
    contact_ranges=[]
    for source in ('wave','high5'):
        selected=[row for row in contacts if row['source_id']==source]
        distances=[row['initial_minimum_contact_distance_m'] for row in selected if row['initial_minimum_contact_distance_m'] is not None]
        contact_ranges.append(dict(source_id=source,
            contact_count=[min(row['initial_contact_count'] for row in selected),max(row['initial_contact_count'] for row in selected)],
            minimum_contact_distance_m=None if not distances else [min(distances),max(distances)],
            no_contact_cases=sum(row['initial_contact_count']==0 for row in selected)))
    peak_curves=[]
    for scenario in ('nominal','combined_minus'):
        identity=scenario+'__selected_wave_a'
        evaluation=read(panel/'evaluations'/(identity+'.json'))
        time_path=panel/'trials'/identity/'state_times.npy'
        times=np.load(time_path,allow_pickle=False)
        accent=evaluation['accent']
        peak_curves.append(dict(scenario_id=scenario,state_times=pinned(time_path),
            times_s=times.tolist(),smoothed_rms_speed_rad_s=accent['smoothed_rms_speed_rad_s'],
            event_times_s=accent['event_times_s'],dominant_time_s=accent['peak_time_s']))
    goal=next(t['goal'] for t in protocol['tasks'] if t['task_id']=='wave_a')
    return dict(schema='pulsemorph.agentic_robustness_posthoc_report.v1',
        inputs=dict(protocol=pinned(panel/'protocol.json'), summary=pinned(panel/'summary.json'),
                    reporter=pinned(__file__), independent_audit=audit_record),
        independent_audit_accepted=None if audit_record is None else audit_record['audit_passed'],
        scenarios=scenarios, task_order=list(TASKS), rows=rows, tasks=tasks,
        baseline_drift=list(baseline.values()), physical_gate=protocol['physical_gate'],
        initial_contact_diagnostics=dict(acceptance_filter=False,calibration_rows=contacts,source_ranges=contact_ranges),
        wave_a_peak_ranking_diagnostic=dict(curves=peak_curves,window_s=protocol['accent_window_s'],
            target_time_s=goal['feature_targets'][4],tolerance_s=protocol['feature_tolerances'][4],
            scope='Posthoc dominant-peak ranking explanation; frozen target failure unchanged. Two events remain; not evidence that the entire dance shifted phase.'),
        scope='Finite deterministic panel of four fixed selected outputs over two sources. Counts are not IID estimates, confidence intervals, probabilities or aesthetic scores.',
        quality_scope='PD target increments and 20ms finite-difference actual joint acceleration diagnostics only; no continuous-acceleration, hardware safety or aesthetic claim, no reranking.',
        model_call_scope='Zero new experimental proposal turns. Engineering agents for development, audit and reporting are outside that method-call budget; total tokens and cost are unknown.',
        primary='Fixed nominal initial center and B0 denominator',
        secondary='Matched perturbed initial center and Bp denominator',
        physical_trials=summary['physical_trials'], new_llm_calls=summary['new_llm_calls'])


def figures(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    from copy import copy
    values = [r[k]['worst_normalized_error'] for r in report['rows'] for k in ('primary','secondary')]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    vmax = max(1.01, max(finite, default=1))
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
    cmap = copy(plt.get_cmap('RdYlGn_r')); cmap.set_bad('#a0a0a0')
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True, constrained_layout=True)
    for axis, estimand, title in zip(axes, ('primary','secondary'),
        ('Primary: fixed nominal center / B0', 'Secondary: matched condition center / Bp')):
        matrix = np.full((13,4), np.nan)
        for i, scenario in enumerate(report['scenarios']):
            for j, task in enumerate(TASKS):
                row = next(r for r in report['rows'] if r['scenario_id']==scenario and r['task_id']==task)
                value = row[estimand]['worst_normalized_error']
                matrix[i,j] = np.nan if value is None else value
                text = 'NA' if value is None else f'{value:.3f}'
                if not row['physical_gate_passed']: text += '\nPHYS FAIL'
                elif not row[estimand]['intent_passed']: text += '\nTARGET FAIL'
                axis.text(j,i,text,ha='center',va='center',fontsize=8)
        im = axis.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
        axis.set_xticks(range(4), TASKS, rotation=25)
        axis.set_yticks(range(13), report['scenarios'])
        axis.set_title(title, fontsize=11)
    fig.colorbar(im, ax=axes, label='max absolute feature error / tolerance (target pass ≤ 1)', shrink=.7)
    fig.suptitle('Fixed-output initial-state sensitivity: 13 declared scenarios\nPhysical gate is additional; no population inference', fontsize=13)
    fig.savefig(output/'normalized_error_heatmaps.png', dpi=180)
    plt.close(fig)
    columns = ['Task', 'Nominal\nprimary', 'Nonzero\nprimary', 'Nonzero\nsecondary', 'Nonzero\nphysical', 'Min root range\n(m)', 'Mean RMSE range\n(rad)']
    cells = []
    for task in report['tasks']:
        nominal, nonzero = task['counts']['nominal'], task['counts']['nonzero']
        cells.append([task['task_id'], f"{nominal['primary_passed']}/1", f"{nonzero['primary_passed']}/12",
            f"{nonzero['secondary_passed']}/12", f"{nonzero['physical_passed']}/12",
            '–'.join(f'{v:.4f}' for v in task['physical_range']['min_root_height_m']),
            '–'.join(f'{v:.4f}' for v in task['physical_range']['mean_joint_rmse_rad'])])
    fig, axis = plt.subplots(figsize=(13,3.2)); axis.axis('off')
    table=axis.table(cellText=cells,colLabels=columns,loc='center',cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.9)
    axis.set_title('Counts split nominal / 12 nonzero; physical ranges cover all 13 cases', pad=12)
    fig.text(.5,.04,'Four related tasks over two sources; no confidence intervals, probability or aesthetic claim.',ha='center',fontsize=9)
    fig.savefig(output/'finite_panel_table.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    diagnostic=report['wave_a_peak_ranking_diagnostic']
    fig,axis=plt.subplots(figsize=(10,4.8),constrained_layout=True)
    target,tolerance=diagnostic['target_time_s'],diagnostic['tolerance_s']
    axis.axvspan(target-tolerance,target+tolerance,color='#87b9dc',alpha=.25,label='Frozen target ± tolerance')
    axis.axvline(target,color='#357ca8',linestyle='--',linewidth=1)
    for curve,color in zip(diagnostic['curves'],('#2274a5','#c45336')):
        times=np.asarray(curve['times_s']); speed=np.asarray(curve['smoothed_rms_speed_rad_s'])
        mask=(times>=1.8)&(times<=3.2)
        axis.plot(times[mask],speed[mask],color=color,label=curve['scenario_id'].replace('_',' '),linewidth=2)
        for event in curve['event_times_s']:
            if 1.8<=event<=3.2:
                index=int(np.argmin(abs(times-event)))
                axis.scatter(times[index],speed[index],color=color,s=30,zorder=3)
        winner=int(np.argmin(abs(times-curve['dominant_time_s'])))
        axis.scatter(times[winner],speed[winner],color=color,marker='*',s=150,zorder=4)
    axis.set(xlim=(1.8,3.2),xlabel='Simulation time (s)',ylabel='Smoothed RMS joint speed (rad/s)',
        title='wave_a: the dominant event changes rank; both peaks remain')
    axis.legend(loc='lower right'); axis.grid(alpha=.18)
    fig.suptitle('Posthoc diagnostic · frozen combined-minus TARGET FAIL remains unchanged',fontsize=10)
    fig.savefig(output/'wave_a_peak_ranking.png',dpi=180); plt.close(fig)


def report(panel, output, audit=None):
    result = collect(panel, audit)
    output=Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    figures(result, output)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    result=report(args.panel,args.output,args.audit)
    print(json.dumps(dict(rows=len(result['rows']), independent_audit_accepted=result['independent_audit_accepted'])))
