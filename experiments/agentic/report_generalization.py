"""Generate a Turkish report only after the complete 98-trial final audit.

python experiments/agentic/report_generalization.py --pilot PILOT_DIRECTORY
    --audit AUDIT_DIRECTORY/audit.json --output NEW_REPORT.md
Read-only inputs; no physics, models, evaluator imports, or plot generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('surrogate_search', 'llm_feedback', 'llm_feedback_off')
LABELS = {'surrogate_search': 'RBF araması', 'llm_feedback': 'LLM + geri bildirim',
          'llm_feedback_off': 'LLM geri bildirim kapalı'}
COUNTS = {'physical_trials': 98, 'control_steps': 19600, 'physics_steps': 392000}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, digits=3):
    if value is None:
        return 'yok'
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite report metric')
    return f'{value:.{digits}f}'


def passed(value):
    return 'evet' if value is True else 'hayır'


def ratios(features):
    return f'{number(features[0])}/{number(features[1])}; {number(features[2])}/{number(features[3])}'


def link(path, output):
    relative = os.path.relpath(Path(path).resolve(), output.parent.resolve()).replace('\\', '/')
    return f'[{Path(path).name}](<{relative}>)'


def select(rows):
    eligible = [r for r in rows if r['integrity_passed'] and r['physical_gate_passed'] and r.get('utility') is not None]
    return min(eligible, key=lambda r: (r['utility'], r['utility_l2'], r['candidate_id'])) if eligible else None


def load_complete(pilot, audit_path):
    protocol = read(pilot / 'protocol.json')
    comparison = read(pilot / 'comparison.json')
    audit = read(audit_path)
    require(audit.get('audit_passed') is True and not audit.get('errors'), 'successful final audit required')
    require(audit.get('original_evidence_unchanged') is True and audit.get('audit_sources_unchanged') is True,
            'audit evidence/source integrity required')
    require(comparison.get('comparison_valid') is True, 'valid completed comparison required')
    require(all(comparison.get(key) == value for key, value in COUNTS.items()), '98 completed physical trials required')
    require(all(audit.get('actual_counts', {}).get(key) == value for key, value in COUNTS.items()), 'audit count mismatch')
    require(audit['actual_counts'].get('individual_model_responses') == 64, '64 audited model responses required')
    require(comparison['protocol_sha256'] == sha(pilot / 'protocol.json'), 'protocol hash mismatch')
    require(len(protocol['tasks']) == len(comparison['tasks']) == 4, 'four tasks required')
    sources = {t['source_id']: t['source_motion'] for t in protocol['tasks']}
    require(len(sources) == len({s['sha256'] for s in sources.values()}) == 2, 'two source groups required')
    for source in sources.values():
        require(sha(ROOT / source['path']) == source['sha256'], 'motion source changed')
    for path, digest in protocol['execution_code_sha256'].items():
        require(sha(ROOT / path) == digest, 'execution code pin changed: ' + path)
    for path, digest in audit.get('source_hashes_after', {}).items():
        require(sha(ROOT / path) == digest, 'audit code pin changed: ' + path)
    canonical = {t['task_id']: t for t in protocol['tasks']}
    rows = []
    for task in comparison['tasks']:
        require(task['goal'] == canonical[task['task_id']]['goal'], 'canonical goal changed')
        require({m['method'] for m in task['methods']} == set(METHODS) and len(task['methods']) == 3, 'method inventory mismatch')
        for method in task['methods']:
            trials = method['trials']
            require(len(trials) == 8, 'eight trials per stream required')
            for index, trial in enumerate(trials):
                identity = f'{task["task_id"]}__{method["method"]}__{index}'
                require(trial['candidate_id'] == identity, 'trial stream identity mismatch')
                audited = audit['trials'][identity]
                for key in ('features', 'utility', 'utility_l2', 'intent_passed', 'physical_gate_passed', 'quality_diagnostics'):
                    require(trial.get(key) == audited.get(key), 'audit metric mismatch: ' + identity + ':' + key)
            selected = select(trials)
            require(method['selected'] == selected, 'stored selector mismatch')
            rows.extend(trials)
    rows.extend(comparison['calibrations'])
    require(len(rows) == len({r['candidate_id'] for r in rows}) == 98, '98 unique completed rows required')
    require({r['candidate_id'] for r in rows} == set(audit['trials']), 'audited trial inventory mismatch')
    for row in rows:
        path = pilot / 'trials' / row['candidate_id'] / 'manifest.json'
        require(sha(path) == row['execution_manifest_sha256'], 'execution manifest changed')
        require(row.get('quality_diagnostics') == audit['trials'][row['candidate_id']].get('quality_diagnostics'),
                'audited baseline/candidate quality mismatch')
    return protocol, comparison, audit, rows


def generate(pilot, audit_path, output):
    pilot, audit_path, output = Path(pilot).resolve(), Path(audit_path).resolve(), Path(output).resolve()
    require(not output.exists(), 'output already exists; refusing overwrite')
    require(output.suffix.lower() == '.md', 'new Markdown output required')
    protocol, comparison, audit, rows = load_complete(pilot, audit_path)
    tasks = {t['task_id']: t for t in protocol['tasks']}
    calibrations = {r['candidate_id'].removeprefix('calibration_'): r for r in comparison['calibrations']}
    manifests = [read(pilot/'trials'/row['candidate_id']/'manifest.json') for row in rows]
    warmups = sum(m.get('warmup_onnx_call_count', 0) for m in manifests)
    warmup_note = ('Her koşuda bir warmup ONNX çağrısı doğrulandı: toplam 98 warmup, 19.600 aktif kontrol çağrısından ayrıdır.'
                  if all(m.get('warmup_onnx_call_count') == 1 for m in manifests)
                  else f'Manifestlerde kayıtlı warmup ONNX çağrıları: {warmups}; aktif kontrol kararlarından ayrıdır.')
    require(all(m.get('control_dt') == .02 for m in manifests), '50 Hz control sampling not confirmed')
    lines = ['# İki kaynakta agentic çok özellikli deney', '',
        'Bu rapor tamamlanmış deney ve geçen bağımsız kayıt auditinden üretilmiştir; yeni fizik veya model çağrısı içermez.', '',
        f'Gerçek bütçe: **{comparison["physical_trials"]} fizik koşusu, {comparison["control_steps"]} kontrol kararı, {comparison["physics_steps"]} MuJoCo adımı**; 96 aday + 2 kaynak kalibrasyonu.', '',
        warmup_note + ' Gerçekleşen kontrol örnekleme aralığı 20 ms (50 Hz) olarak doğrulandı.', '',
        'Dört görev, iki motion dosyası/kaynak grubu üzerinde ilişkilidir; aynı G1 beden, pretrained motor ve generated kaynak ailesini kullanır. Cue yerleşimi fizik öncesi kaynak bilgisiyle hazırlanmıştır; bu saf kör kaynak değerlendirmesi veya bağımsız insan genellemesi değildir.', '',
        '## Görevler ve hedefler', '',
        'Oran sırası L_A/R_A; L_B/R_B (sol/sağ, bölüm A; sol/sağ, bölüm B). Zaman hedefleri gerçek waveform analizinden gelen kanonik değerlerdir.', '',
        '| Görev | Kaynak | Hedef oranlar | Hedef vurgu s |', '|---|---|---|---:|']
    for task in protocol['tasks']:
        target = task['goal']['feature_targets']
        lines.append(f'| {task["task_id"]} | {task["source_id"]} | {ratios(target)} | {number(target[4],6)} |')
    tolerance = protocol['feature_tolerances']
    lines += ['', f'Mutlak toleranslar: L_A={number(tolerance[0])}, R_A={number(tolerance[1])}, L_B={number(tolerance[2])}, R_B={number(tolerance[3])}; vurgu zamanı ±{number(tolerance[4]*1000,1)} ms.', '',
        '## Seçilen adaylar', '',
        'Oranlar gerçekleşen fiziksel eklem hareketinin aynı kaynağın değişmemiş fizik kalibrasyonuna oranıdır. Sıra sol-A, sağ-A, sol-B, sağ-B; zaman hatası baskın hareket vurgusu ile ölçülen müzik cue’sunun mutlak farkıdır.', '',
        '| Görev | Yöntem | Aday indeksi | L_A/R_A; L_B/R_B | Zaman hatası ms | Tüm hedefler | En kötü normalize hata |',
        '|---|---|---:|---|---:|---|---:|']
    selected_rows = []
    for task in comparison['tasks']:
        for method in task['methods']:
            selected = method['selected']
            if selected is None:
                lines.append(f'| {task["task_id"]} | {LABELS[method["method"]]} | yok | — | — | hayır | — |')
                continue
            features = selected.get('features')
            require(isinstance(features, list) and len(features) == 5, 'selected feature vector required')
            target = tasks[task['task_id']]['goal']['feature_targets']
            timing = abs(features[4] - target[4]) * 1000 if features[4] is not None else None
            index = int(selected['candidate_id'].rsplit('__',1)[1])
            lines.append(f'| {task["task_id"]} | {LABELS[method["method"]]} | {index} | {ratios(features)} | {number(timing,1)} | {passed(selected["intent_passed"])} | {number(selected["utility"])} |')
            selected_rows.append((task, method, selected))
    lines += ['', 'Aday indeksi 0–7 aralığındadır; 0 değişmemiş anchor’dır. Tam aday kimlikleri bağlantılı comparison.json içinde korunur. Seçim fizik kapısından geçen adaylar arasında en büyük mutlak tolerans hatası, sonra normalize hata kareleri toplamı ve aday kimliği sırasındadır. Bu selector genel dans kalitesi puanı değildir; kalite diagnostikleri seçimi yeniden sıralamaz.', '',
              '## Yöntem ve bütçe özeti', '',
              '| Yöntem | Seçilen adayla başarılı görev / 4 | Başarılı aday / 32 | Fizik kapısı / 32 | İlk başarılı deneme (1–8, anchor dahil) |',
              '|---|---:|---:|---:|---|']
    for name in METHODS:
        streams = [(t, next(m for m in t['methods'] if m['method'] == name)) for t in comparison['tasks']]
        trials = [r for _, m in streams for r in m['trials']]
        successes = sum(bool(m['selected'] and m['selected']['intent_passed']) for _, m in streams)
        first = []
        for task, method in streams:
            successful = next((i+1 for i,r in enumerate(method['trials']) if r['intent_passed']), None)
            first.append(task['task_id'] + ': ' + (str(successful) if successful is not None else 'yok'))
        lines.append(f'| {LABELS[name]} | {successes}/4 | {sum(r["intent_passed"] is True for r in trials)}/32 | {sum(r["physical_gate_passed"] is True for r in trials)}/32 | {"; ".join(first)} |')
    lines += ['', f'Tüm koşularda fizik kapısı: {sum(r["physical_gate_passed"] is True for r in rows)}/98; kayıt bütünlüğü: {sum(r["integrity_passed"] is True for r in rows)}/98. İki kalibrasyon intent yarışmasına dahil değildir.', '',
        'Her LLM yönteminde her görev için sekiz ayrı model yanıtı/turu vardır: feedback 32, feedback-off 32, toplam 64. Çağrı sayıları eşleştirilmiştir; token, fatura, bağlam uzunluğu veya izole model gecikmesi eşitliği gösterilmemiştir. OFF için 32 önerinin tamamı aday fiziğinden önce taahhüt edilmiştir; feedback yalnız kendi önceki sonuçlarını görür. Gözlem kısıtları prompt ve ayrı fork/agent akışlarına dayanır; işletim sistemi düzeyinde sandbox izolasyonu iddiası değildir.', '',
        '## Ayrı kalite diagnostikleri', '',
        'Aşağıda her hücre seçilen aday / aynı kaynak kalibrasyonu değeridir. Önceden tanımlanmış kalite kabul eşiği yoktur; yüksek veya düşük değer tek başına iyi dans anlamına gelmez.', '',
        '| Görev / yöntem | PD hedef adımı RMS / max rad | Gerçek hızdan FD ivme RMS / max rad/s² | Root displacement farkı RMS / son m |',
        '|---|---|---|---|']
    metrics = [('applied_pd_target_step_rms_rad', 'applied_pd_target_step_max_abs_rad'),
               ('joint_acceleration_finite_difference_rms_rad_s2', 'joint_acceleration_finite_difference_max_abs_rad_s2'),
               ('root_displacement_difference_rms_m', 'root_displacement_difference_final_m')]
    for task, method, selected in selected_rows:
        baseline = calibrations[task['source_id']]['quality_diagnostics']
        diagnostics = selected['quality_diagnostics']
        cells = ['; '.join(number(diagnostics[key]) + ' / ' + number(baseline[key]) for key in pair) for pair in metrics]
        lines.append(f'| {task["task_id"]} / {LABELS[method["method"]]} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'PD adımı uygulanmış hedef farkıdır. İvme 20 ms kontrol sınırlarında gerçekleşen qvel farkından hesaplanır; adım içi ivmeyi ölçmez. Root farkı unwarped kaynak ile source-world displacement karşılaştırmasıdır; root rota kontrolü veya kayma mesafesi değildir. Temas/slip, insan beğenisi ve algısal ifade kalite kabulü ölçülmemiştir.', '',
        '## Kapsam ve köken', '',
        'Bütün yöntemlere aynı özgün kanonik hedef verildi; ayrı semantik aşama yoktur. Model gözlemi waveform analizinden sayısal müzik özellikleri alır; ham PCM’i duyduğu veya genel müzik algısını çözdüğü iddia edilmez. Hedeflerin fiziksel erişilebilirliği bilinmemektedir; sekiz adayda başarısızlık görevin olanaksızlığını kanıtlamaz.', '',
        'Bu deney canlı recovery, eğitimle ağırlık geliştirme, güvenli donanım kontrolü, üretim kabulü veya genel dans kalitesi sonucu vermez. Eski v1’in 2/2 geliştirme görevi sonucu yalnız tarihsel bağlamdır; yeni görevler ve kaynaklarla istatistiksel karşılaştırma değildir.', '',
        f'Protokol: {link(pilot/"protocol.json",output)}; SHA256 `{sha(pilot/"protocol.json")}`. Karşılaştırma: {link(pilot/"comparison.json",output)}; SHA256 `{sha(pilot/"comparison.json")}`.', '',
        f'Bağımsız audit: {link(audit_path,output)}; SHA256 `{sha(audit_path)}`; audit_passed=true. Kaynak kod pinleri protokolün execution_code_sha256 alanında; audit kaynak pinleri audit dosyasında kayıtlıdır ve rapor üretiminde yeniden kontrol edilir.', '',
        '| Kaynak grubu | Motion yolu | SHA256 |', '|---|---|---|']
    sources = {task['source_id']: task['source_motion'] for task in protocol['tasks']}
    for name, source in sorted(sources.items()):
        lines.append(f'| {name} | {link(ROOT/source["path"],output)} | `{source["sha256"]}` |')
    video_lines = []
    for task in comparison['tasks']:
        video = ROOT/'artifacts/media/agentic/generalization_music_v1'/task['task_id']/'comparison.mp4'
        if video.is_file():
            video_lines.append(f'- {task["task_id"]}: {link(video,output)}')
    if video_lines:
        lines += ['', '## Videolar', '', 'Videolar gerçekleşen kayıtları incelemek içindir; ayrı insan kalite değerlendirmesi değildir.', ''] + video_lines
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(generate(args.pilot, args.audit, args.output))


if __name__ == '__main__':
    main()
