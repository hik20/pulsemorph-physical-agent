# Sabit seçilmiş hareketlerin başlangıç durumuna hassasiyeti

27 Ekim 2026. Dört önceki feedback seçiminin sabit parametreleriyle, önceden dondurulmuş 13 senaryo tamamlandı. [Bağımsız audit](../../artifacts/runs/agentic/robustness_initial_state_v2_audit_v1/audit.json) `audit_passed=true`, errors=[]: 78 fizik koşusu, 26 calibration ve 52 seçilmiş çıktı denetlendi. Altı nominal koşunun özgün trace'lere dtype/shape/byte eşitliği, nonzero koşulardan önce geçti. Bu çalışma önceki development karşılaştırmasının 4/4 sonucunu değiştirmez; farklı gerçek başlangıç durumları altında aynı dört çıktıyı ayrıca ölçer.

[Dondurulmuş protokol](../../artifacts/runs/agentic/robustness_initial_state_v2/protocol.json), [ham özet](../../artifacts/runs/agentic/robustness_initial_state_v2/summary.json) ve [ayrıntılı posthoc rapor](../../artifacts/runs/agentic/robustness_initial_state_v2_report_v1/report.json) hash ile bağlıdır. Posthoc rapor modeli veya fiziği çalıştırmaz, hedefleri yeniden seçmez. [76 test ve 11 alt testin son doğrulama logu](../../artifacts/runs/agentic/robustness_verification_v2/final_pytest.log) ve [24.38s/exit0 doğrulama kaydı](../../artifacts/runs/agentic/robustness_verification_v2/test_run.json) bağımsız fizik audit'inden ayrı kanıttır. Üç PNG'nin tamamı açılarak etiket/kırpılma/yerleşim açısından incelendi; [görsel QA](../../artifacts/runs/agentic/robustness_initial_state_v2_report_v1/visual_qa.json) boyut ve hash'lerini taşır. Bu teknik QA insan beğenisi veya hareket kalitesi kabulü değildir.

## Sabit panelde gözlenen sonuç

Panel: nominal + joint q ±.01 rad, joint dq ±.05 rad/s, dünya vx/vy ayrı ayrı ±.03 m/s, yaw ±1 derece ve bu beş bileşenin combined ± çifti. Eklem yönleri isim-hash ile deterministiktir. Kaynak/görev başına yeni yön veya parametre seçilmedi. Yaw, normal upstream heading autoalignment altında heading-equivariance tanısıdır; mutlak heading hata testi değildir.

Birincil ölçüm fresh nominal calibration'ın başlangıç merkezi ve B0 RMS paydasını sabitler; byte eşitliği bunun özgün hedef temelini koruduğunu doğrular. İkincil ölçüm eşlenmiş perturbed calibration'ın kendi başlangıç merkezi ve Bp paydasını kullanır. Başarı beş hedef ve mevcut fizik kapısının birlikte geçmesidir. Calibration fizik kapısına göre seçilmiş aday filtrelenmedi.

| Görev | Nominal birincil | 12 nonzero birincil | 12 nonzero ikincil | Nonzero fizik |
|---|---:|---:|---:|---:|
| wave_a | 1/1 | 11/12 | 11/12 | 12/12 |
| wave_b | 1/1 | 12/12 | 12/12 | 12/12 |
| high5_a | 1/1 | 12/12 | 12/12 | 12/12 |
| high5_b | 1/1 | 4/12 | 5/12 | 12/12 |

Toplam nonzero sayımı birincil 39/48, ikincil 40/48; nominal dört çıktı her ikisinde 4/4'tür. Bunlar bağımsız rastgele deneme başarı olasılığı veya güven aralığı değildir. Eksik baskın accent olayı yoktur: 52 seçilmiş değerlendirmede accent available=true.

![Birincil ve ikincil normalize hata panelleri](../../artifacts/runs/agentic/robustness_initial_state_v2_report_v1/normalized_error_heatmaps.png)

## Bütün birincil başarısızlıklar

Worst = max(abs(feature error)/tolerance); hedef sınırı ≤1. Aşağıdaki high5_b satırlarının tamamında tek başarısız boyut sol-B RMS oranıdır: hedef .75±.08, üst sınır .83. Diğer dört hedef geçer. Wave_a satırında dört RMS hedefi geçer, accent hedefi geçmez. Bu dokuz satırın tamamı fizik kapısını geçer.

| Görev / senaryo | Başarısız ölçüm | Worst | Kabul sınırının dışındaki fark |
|---|---:|---:|---:|
| high5_b / joint_position_minus | sol-B .866552165 | 1.456902 | +.036552165 oran |
| high5_b / joint_velocity_plus | sol-B .831360579 | 1.017007 | +.001360579 oran |
| high5_b / joint_velocity_minus | sol-B .833625292 | 1.045316 | +.003625292 oran |
| high5_b / root_vx_plus | sol-B .843953544 | 1.174419 | +.013953544 oran |
| high5_b / root_vx_minus | sol-B .831059302 | 1.013241 | +.001059302 oran |
| high5_b / root_vy_plus | sol-B .852257590 | 1.278220 | +.022257590 oran |
| high5_b / combined_plus | sol-B .846535135 | 1.206689 | +.016535135 oran |
| high5_b / combined_minus | sol-B .861159344 | 1.389492 | +.031159344 oran |
| wave_a / combined_minus | accent 2.400000 s | 5.932200 | alt sınırdan .493219955 s erken |

İki estimand'ın pass/fail farkı yalnız high5_b / joint_velocity_plus'tır: birincil sol-B .831360579 ile başarısız; ikincil .829550955 ile geçer, üst sınıra yalnız .000449045 oran marjı kalır. Bu geçiş birincil başarısızlığı silmez. İkincildeki diğer yedi high5_b başarısızlığı ve wave_a combined_minus başarısızlığı korunur.

Baseline drift aralıkları, her kaynakta 13 senaryonun dört kol/pencere boyutu boyunca: wave için Bcondition-about-c0/B0 .973868–1.050602, Bp/B0 .975443–1.048200; high5 için .992291–1.029262 ve .962585–1.068751. Rapor JSON'u her boyutun merkez farkını, RMS'ini, signed hatasını ve kabul marjını ayrı taşır. Hareketli merkez/paydanın ortak sapmayı gizleyebilmesi nedeniyle birincil temel sabit tutulur.

## Wave_a başarısızlığı: iki tepenin sırası değişiyor

Hedef 2.993219955±.1 s'dir. Nominal baskın tepe 2.9 s, combined_minus baskın tepe 2.4 s: nominal hata −.093219955 s, perturbed hata −.593219955 s. Ancak her iki olay her iki eğride de vardır. 2.4/2.9 s smoothed RMS hızları nominalde .448704690/.453224224, combined_minus'ta .441639743/.440806796 rad/s'dir. Küçük hız farkı dominant kazananı değiştirir. **Bütün dansın .5 s faz kaydığı sonucu çıkarılmaz.** Donmuş dominant-event hedef başarısızlığı değişmez; bu grafik posthoc açıklayıcı tanıdır.

![Wave_a baskın tepe sıralaması](../../artifacts/runs/agentic/robustness_initial_state_v2_report_v1/wave_a_peak_ranking.png)

## Fizik kapısı ve kapsamı

Mevcut kapı minimum root≥.60 m ve mean timestamp-aligned joint RMSE≤.35 rad'dır. Aşağıdaki aralıklar **nominal+12 nonzero, tüm13** seçilmiş koşulu kapsar; sayım tablolarındaki nonzero paydasıyla karıştırılmaz.

| Görev | Minimum root aralığı m | Mean joint RMSE aralığı rad | Max PD adımı aralığı rad | Max 20ms FD ivme aralığı rad/s² |
|---|---:|---:|---:|---:|
| wave_a | .787629–.788739 | .156220–.162726 | .670649–.918412 | 116.535–544.678 |
| wave_b | .789246–.789564 | .155221–.156245 | .670609–.839018 | 114.468–217.258 |
| high5_a | .784896–.787762 | .135222–.136989 | .747972–.866984 | 265.515–320.139 |
| high5_b | .785557–.786793 | .129679–.131108 | .862731–.882621 | 233.085–324.932 |

PD target adımı gerçekleşen beden ivmesi değildir. İvme 20ms örnekli finite difference'dır; sürekli tepe, donanım güvenliği veya estetik sonucu vermez. Yeni eşik veya bu tanılara göre sıralama yapılmadı. 26 calibration başlangıcının tamamında contact count0, minimum contact distance=null'dır; bu yalnız başlangıç temas tanısıdır, bütün koşuda temas yokluğu veya temas kalitesi sonucu değildir. Clip, root yükseltme veya yönü yeniden seçme uygulanmadı.

## Karantina ve maliyet

İlk V1 paneli üçüncü nominal fizik koşusundan sonra NumPy bool_ JSON serialization hatasıyla durdu. [V1 karantinası](../../artifacts/runs/agentic/robustness_initial_state_v1_quarantine_v1/summary.json) üç fizik, 600 kontrol, 12.000 fizik adımı, sıfır nonzero koşu ve sıfır yeni öneri turunu kaydeder. Eval JSON'u yayımlanmadı; altılı nominal kapı tamamlanmadı. [Salt okunur üç-trace audit'i](../../artifacts/runs/agentic/robustness_initial_state_v1_quarantine_v1/raw_trace_audit.json) raw integrity/input pins/byte parity'yi doğruladı; tamamlanmış hedef değerlendirmesi sayılmaz. V1 kodu ve kanıtı korunmuştur.

V2 yalnız yeni sürümlü JSON-native evaluator wrapper ve bunun yeni plan/driver/audit bağını kullanır; adaylar, senaryolar, referanslar, hedefler ve kapılar değişmedi. Geçerli V2 paneli 78 fizik/15.600 kontrol/312.000 adım; bu sağlamlık işinin V1+V2 toplamı **81 fizik/16.200 kontrol/324.000 adım/0 yeni deneysel LLM öneri turu**dur. Önceki development aramasının maliyeti bu toplamda yoktur. Mühendislik, audit ve raporlama agentları deneysel öneri bütçesine dahil değildir; toplam token/ücret bilinmiyor.

Sonuç iki kaynak, aynı beden/motor ve dört ilişkili development seçiminin bu deterministik yerel paneldeki hassasiyetidir. Yeni kaynak genellemesi, rastgele seed dağılımı, ağırlık öğrenimi, recovery, algılanan ifade/insan beğenisi veya ürün kabulü ölçülmedi. Training admission=false, kalite=unassessed. Aynı çıktının nominal geçmesi özellikle high5_b'de bu yerel perturbation panelinde hedefi koruyacağını garanti etmedi.
