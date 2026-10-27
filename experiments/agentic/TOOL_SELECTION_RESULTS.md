# İki clock aracıyla development karşılaştırması — v2 sonuç raporu

26–27 Ekim 2026. Son güncelleme:27 Ekim2026 (Europe/Istanbul).

**Durum: 100 fizik /64 yeni LLM çağrısı tamamlandı; metadata projection üzerinden bağımsız audit geçti.** Feedback 4/4, feedback-off 1/4, sayısal arama 0/4 görevde beş hedefi birlikte karşılıyor. Bunlar iki kaynaklı development panelinin, tüm sekiz adaydan sabit selector ile seçilmiş sonuçlarıdır. Özgün runtime'ın strict G1 metadata kabulü geçmedi; aşağıda açıklanan sayısal doğrulama ve sonradan yapılan iki strict-compatible replay bu durumdan ayrı tutulur. Bu rapor güncellemesi yeni fizik, model çağrısı veya test çalıştırmadı.

## Soru, kaynaklar ve deney kimliği

Soru: aynı iki düzenleme aracı ve görev başına sekiz gerçek fizik denemesi verildiğinde, sayısal arama ile fizik geri bildirimi açık/kapalı LLM'nin araç ve parametre seçimleri bu dört geliştirme görevinde nasıl sonuçlanıyor? Araç seçimi ve beş sürekli parametre birlikte ölçülür; doğru araca ilişkin oracle etiketi yoktur. Genel LLM üstünlüğü, öğrenilmiş yeni beceri veya en güçlü olası baseline'a üstünlük bu deneyin tanımı değildir.

[Dondurulmuş v2 protokolü](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2/protocol.json) SHA256: `bb3c05bf381b59445ada289d689eaa7ce3086bbfb2612829e4489618de96ae2c`. Dört görev `wave_a`, `wave_b`, `high5_a`, `high5_b`; kaynaklar `output_wave.motion` ve `output_high5.motion`. Kaynak dosyaları, Türkçe talimatlar, canonical hedefler ve ses waveform byte'ları önceki `generalization_music_v1` görevlerinden alınmıştır. Sesin yeni dizindeki yolu değişmiştir; içerik ve ölçülmüş cue korunmuştur. Eski aday etiketi, calibration, response veya fit edilmiş model v2'ye alınmamıştır. Önceki geliştirme bilgisinin araç ve arama tasarımına etkisi açıktır.

Bu, **iki daha önce kullanılan kaynakta dört ilişkili development görevidir**. Dört bağımsız kaynak, yeni insan, yeni beden, yeni motor veya held-out genelleme paneli değildir. Aynı Kimodo-generated aile, G1 beden ve sabit pretrained motor kullanılır. [Statik kaynak incelemesine](D:/projects/dance/design/agentic/TOOL_SELECTION_SOURCE_REVIEW.md) göre mevcut trusted allowlist içinde süre koşulunu karşılayan tek yeni kaynak walk'tır; jumpjack 3.966667 s ile gerekli 4.14 s'yi desteklemez. Dört yeni kaynaklı ana panel hedefi korunur ve **kaynak yetersizliği nedeniyle engelli** kalır. Development karşılaştırması bu hedefin yerine geçirilmez.

## Ortak bilgi ve yöntemler

Her yöntem aynı canonical beş hedefi, waveform extractor gözlemini, kaynak kimliğini, parametre sınırlarını ve iki araç açıklamasını alır. Canonical sayılar zaten verildiğinden bu deney Türkçe ayrıştırma, ham ses anlama veya yeni koreografi üretme puanı değildir. Ses gözlemi dört saniyenin tamamını kapsayan offline, zero-phase analizdir; canlı/gelecek ses görmeyen bir müzik sistemi sınanmıyor.

| Yöntem | Aday bilgisi | Her görev için bütçe |
|---|---|---|
| `surrogate_search` | Ortak fresh calibration ve yalnız kendi gerçekleşmiş sonuçları | 8 aday/fizik; 0 LLM çağrısı |
| `llm_feedback` | Aynı ortak bilgi ve kendi önceki öneri/ölçümleri | 8 ayrı `gpt-6.1-sol` öneri çağrısı, 8 aday/fizik |
| `llm_feedback_off` | Aynı ortak bilgi ve kendi önceki önerileri; aday sonucu yok | 8 ayrı `gpt-6.1-sol` öneri çağrısı, 8 aday/fizik |

LLM koşullarının model/reasoning ayarı aynıdır; bağlamlar ayrıdır, kardeş sonuçları verilmez. Dosya erişimi açık talimatla sınırlandırılmıştır; OS sandbox ile zorlanmış izolasyon iddiası yoktur. V2'nin 32 feedback-off önerisi bütün aday fiziğinden önce yeni çağrılarla taahhüt edilmiştir. Çağrı receipt'leri orchestrator tarafından gözlenen gerçek collaboration turlarını kaydeder; sağlayıcının token veya faturalama makbuzu değildir. Feedback kolunun 32 ayrı çağrısı ve adayları da tamamlandı; toplam 64 yeni receipt kaydedildi. Eski v1 çağrıları yeni 64 çağrı bütçesine sayılmaz.

Sayısal optimizer'ın dondurulmuş sırası global seed, local seed, global maximin, local maximin, model seçimi, maximin exploration, model seçimi, maximin exploration'dır. Son dört seçim daha az denenmiş aracı öncelediği için tamamlanmış sekiz slotta global/local dağılımı 4/4 olur. Her araç için ayrı RBF ve sabit deterministik aday havuzu kullanılır; ortak identity calibration tek ölçüm kökenidir, iki bağımsız örnek sayılmaz. Tahmini residual fiziksel ölçüm değildir. LLM sekiz slotun tamamında araç dağılımını serbest seçer. Böylece erişilebilir araçlar ve fizik bütçesi eşittir; arama politikaları veya keşif zorunluluğu aynı değildir. Seçilme sıklığı tek başına doğru araç seçimi veya mekanizma kanıtı olmaz.

## Araçlar, calibration ve kapılar

İki araç `global_clock_v1` ve `local_clock_1p4_2p8_v1`'dir. Her ikisinde dört gain [.35,1.60], shift [−.30,.30] s; kol displacement gain'leri aynı [.4,2.0] ve [2.0,3.6] s sin² zarflarını kullanır. Global clock 0–4 s boyunca, local clock yalnız 1.4–2.8 s içinde kaynak saatini sin² biçiminde kaydırır; dışında identity'dir. Eklem hızları analitik chain rule içerir. Aynı kaynak, ortak MotionReference array verileri, pretrained ONNX, PD ve MuJoCo yürütmesi kullanılır. Özgün 100 koşuda ortak referans yalnız generic `validate` + `sample` yoluyla tüketilmiştir; strict G1 metadata admission bu yürütmelerde çağrılmamıştır. LLM'nin önerisi doğrudan tork veya fiziksel kabul değildir; ağırlıklar güncellenmez.

Her kaynak için global ve local identity olmak üzere **dört yeni calibration koşusu** yapıldı. [Calibration parity kaydı](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2/calibration_parity.json) iki kaynakta qpos/qvel/uygulanan PD hedef byte'ları, ortak q/dq/torso ve gerçek policy input array'leri ile tam preexecution pin uyumunun geçtiğini kaydeder. Global calibration tek ortak oran paydasıdır; local koşu ayrıca sayılan parity kontrolüdür. İlk aday zorunlu identity değildir.

V2 planner baseline'ı yalnız calibration kimliği/parametre/araç, gerçek özellikler, integrity/fizik durumu, RMSE/root yüksekliği ve varsa ölçülen accent alanlarını içerir. Hedefe bağlı signed hata, pass bitleri, intent, utility/L2 veya wall time içermez. [İlk observation kontrolü](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2/initial_observation_check.json), aday fiziği başlamadan önce sekiz başlangıç observation'ında A/B baseline'larının aynı, canonical hedeflerin farklı, hedefe bağlı alanların ve önceki aday sonuçlarının yok olduğunu kaydeder. Kendi aday geçmişinin puanları ise kendi güncel göreviyle hesaplanır.

| Ölçüm/karar | Sabit kural |
|---|---|
| Dört kol özelliği | Gerçekleşmiş eklemlerin başlangıç postürüne göre RMS'i / aynı kaynak global calibration RMS'i; sol/sağ A=.8–1.6 s, B=2.4–3.2 s |
| Accent | Gerçekleşmiş joint-speed RMS proxy'si; 1.8–3.2 s içinde uygun en güçlü peak, .15 rad/s prominence, 5 örnek smoothing; eksik peak başarısız hedef olarak kalır |
| Hedef toleransları | Kol oranlarında [.08,.08,.08,.08], accent zamanında .10 s |
| Fizik kapısı | Minimum root yüksekliği ≥.60 m ve mean joint RMSE ≤.35 rad; ayrıca integrity şartı |
| Seçici | Integrity+fizik kapısından geçenlerden en küçük en-kötü mutlak hata/tolerans; eşitlikte normalize kare-toplam, sonra candidate ID |
| Başarı | Seçilmiş adayın beş hedefin tamamını tolerans içinde karşılaması |
| Slot muhasebesi | Her yöntem/görev 8 slot; başarıda erken durma veya ek confirmation yok. Schema/model/runtime/integrity hatası kanıtı koruyarak tüm deneyi durdurur; ücretsiz repair/retry/replacement yok. Fizik kapısı veya hedef başarısızlığı ölçülmüş aday olarak kalır |

Gain istenen fiziksel RMS oranıyla özdeş değildir. Sayısal fizik kapısı kapsamlı denge, recovery veya donanım güvenliği belgesi değildir. Burada kinematik oynatma fiziksel kontrol sayılmıyor; gerçek motor yürütmesi ayrıca kaydedilir.

## Tamamlanmış v2 bütçesi ve kayıtlı sonuçlar

[Karşılaştırma kaydı](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2/comparison.json) `comparison_valid=true` bildirir: 96 aday +4 calibration =**100 gerçek fizik koşusu, 20.000 kontrol, 400.000 fizik adımı ve 64 yeni LLM öneri turu**. Bütün 100 kayıtlı koşu integrity ve fizik kapısını geçti. Bu driver sonucu bağımsız audit kabulüyle özdeş değildir.

Aşağıda S=sayısal, F=feedback, O=feedback-off; G=global, L=local. `idx` sıfır tabanlı candidate ID son ekidir. Özellik sırası sol A, sağ A, sol B, sağ B, accent saniyesi. `worst` en kötü mutlak hata/toleranstır; ≤1 beşli hedef başarısıdır. Yuvarlama yalnız sunum içindir; selector tam hassasiyetle çalıştı.

| Görev/yöntem | idx/araç | Gerçek beş özellik | worst | RMSE rad / min root m | Başarı |
|---|---|---|---:|---|---|
| wave_a / S | 6/G | [0.929989, 1.045361, 1.108646, 0.951266, 2.900000] | 1.932985 | 0.155646 / 0.788658 | hayır |
| wave_a / F | 7/G | [0.820774, 1.180929, 1.201375, 0.824623, 2.900000] | 0.932200 | 0.159437 / 0.788685 | evet |
| wave_a / O | 1/G | [0.945198, 1.107014, 1.203258, 0.841986, 2.900000] | 1.814977 | 0.156231 / 0.789479 | hayır |
| wave_b / S | 4/G | [0.996479, 0.647490, 0.835296, 1.253412, 2.900000] | 2.229252 | 0.161611 / 0.789480 | hayır |
| wave_b / F | 7/L | [1.100000, 0.750000, 0.750000, 1.100000, 3.120000] | 0.029252 | 0.155652 / 0.789422 | evet |
| wave_b / O | 4/L | [1.112230, 0.783886, 0.778522, 1.127524, 3.100000] | 0.423581 | 0.156389 / 0.789153 | evet |
| high5_a / S | 1/L | [0.860650, 1.135681, 1.248142, 1.006308, 2.160000] | 2.578844 | 0.132363 / 0.786612 | hayır |
| high5_a / F | 6/L | [0.800088, 1.200153, 1.200251, 0.799498, 2.180000] | 0.049206 | 0.136277 / 0.786817 | evet |
| high5_a / O | 5/L | [0.791686, 1.193849, 1.306252, 0.949994, 2.180000] | 1.874921 | 0.134065 / 0.787066 | hayır |
| high5_b / S | 6/G | [1.249562, 0.559769, 0.620035, 0.990977, 2.140000] | 2.377882 | 0.124725 / 0.786707 | hayır |
| high5_b / F | 6/L | [1.135809, 0.743473, 0.829561, 1.057013, 2.220000] | 0.994510 | 0.130095 / 0.786086 | evet |
| high5_b / O | 5/L | [1.125296, 0.724440, 0.991764, 1.203162, 2.220000] | 3.022056 | 0.129989 / 0.785712 | hayır |

Tam ID her satırda `{görev}__{method}__{idx}` biçimindedir; tam parametre, signed hata, pass bitleri ve köken hash’leri comparison/evaluation kayıtlarında saklıdır.

| Görev/yöntem | G/L aday sayısı | İlk başarılı slot (1 tabanlı) | Prefix-best başarı, bütçe 1→8 |
|---|---|---:|---|
| wave_a / S | 4/4 | yok | 00000000 |
| wave_a / F | 8/0 | 3 | 00111111 |
| wave_a / O | 6/2 | yok | 00000000 |
| wave_b / S | 4/4 | yok | 00000000 |
| wave_b / F | 0/8 | 1 | 11111111 |
| wave_b / O | 1/7 | 1 | 11111111 |
| high5_a / S | 4/4 | yok | 00000000 |
| high5_a / F | 1/7 | 3 | 00111111 |
| high5_a / O | 4/4 | yok | 00000000 |
| high5_b / S | 4/4 | yok | 00000000 |
| high5_b / F | 1/7 | 7 | 00000011 |
| high5_b / O | 5/3 | yok | 00000000 |

Her prefix için bütün o ana kadarki geçerli adaylara aynı selector uygulandı; 1=seçilmiş aday beş hedefi karşılıyor. Toplam başarılı görev sayısı F için bütçe1..8 boyunca **[1,1,3,3,3,3,4,4]**, O için **[1,1,1,1,1,1,1,1]**, S için sıfırdır. Başarıda durulmadı. High5_b F ilk geçişi yedinci slotta buldu; sekizinci/son aday yerine `idx6` seçildi.

Wave_a F seçimi G/shift−.30 s; accent hatası −93.219955 ms ile .10 s sınırı içindedir. Wave_b F seçimi L/shift+.20 s; kol oranları hedeflere yaklaşık 2×10⁻⁷ içinde, accent hatası −2.925170 ms’dir. High5_a F seçimi L/shift−.23 s; accent hatası +4.920635 ms’dir. Bunlar ölçülen aday sonuçlarıdır; sıfır-shift teorik araç farkı veya bilinen doğru-araç etiketi çıkarılmadı.

**High5_b F tanığı dar marjlıdır:** `high5_b__llm_feedback__6`, L, parametreler `[1.20,.63,.35,.90,−.30]`; sol B oranı **.8295607673601977** ve izinli üst sınır .83 olduğundan kalan marj yalnız **.0004392326398023**’tür. Worst=.994509592; accent hatası −74.807256 ms. Ana 100 koşuluk karşılaştırmada confirmation yoktur. Sonradan aynı parametrelerle yeni strict-compatible kodda bir ayrı fresh-reset replay yapıldı ve tam fizik iz eşitliği doğrulandı; bu iki deterministik kod yolunun uyum kanıtıdır, gürültü/başlangıç perturbasyonu karşısında sağlamlık değildir. Dar marj korunur. Eski capability confirmation farklı parametre ve bütçededir, bu adaya ödünç doğrulama olmaz.

### Ortak metadata hatası, projection audit'i ve iki compatibility replay

İlk [raw audit](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2_audit_v1/audit.json) standart dışı üst düzey `control_tool` ve `experiment_kind` alanlarında strict G1 assembler tarafından reddedildi. Özgün common package'lar generic MotionReference validation ve sampling'den geçmişti; bu, strict G1 admission ile aynı kontrol değildir. Donmuş runtime/package/sonuçlar yerinde değiştirilmedi; başarısız audit de korundu.

Her 100 common reference için ayrı metadata-only compatible projection oluşturuldu: yalnız bu iki standart dışı üst düzey alan çıkarıldı; araç/model bilgisi standart `producer` nesnesinde korunur, deney bilgisi dış manifesttedir. Dört array'in dtype/shape/byte verileri değişmedi. [100 uyumlu hareket paketinin indeksi](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2_compatible_refs_v1/index.json) özgün ve uyumlu dosyaları hash'leriyle bağlar. [Tam projection audit'i](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v2_projection_audit_v1/audit.json) `audit_passed=true`, errors=[]:100 koşunun bağımsız ölçüm/score hesapları,20.000 kararın strict assembler/reference input karşılaştırması,400.000 kayıtlı fizik adımı,64 response/receipt, calibration, optimizer replay ve selector kontrolleri geçti. Her satırda gerçek reference-input hatası0; en büyük gerçek torso/FK açısal farkı4.6892345742347425×10⁻⁸rad. Projection index SHA256 `7b1c2b141544eb01971931bd12fd090da1cea239c51ff3ae9dec2f00590edbbf`. Audit `original_strict_g1_admission=false`, `compatible_projection_required=true`, `projection_raw_arrays_unchanged=true` olarak açık kayıt tutar. Özgün kanıt, export, başarısız raw audit ve audit kodu değişmedi.

Bu, değişmeyen referans array'leri ve gerçek fizik kayıtlarının standart metadata projection'ıyla strict arayüzde sayısal olarak uyuştuğunu gösterir; özgün 100 koşunun gerçekte strict admission çağırdığı anlamına gelmez. Projection yeni fizik veya yeni skor üretmez; mevcut skorları bağımsız yeniden hesaplayıp doğrular.

Ayrı uyum runtime'ı standard metadata'yı kaydettikten ve yeniden yükledikten sonra kontrolcü öncesi gerçek strict G1 assembler'ı0/3.98s'de çağırır (`strict_g1_admission_passed=true`, preflight_count2). İki fresh-reset4s koşusu, yeni model çağrısı veya optimizasyon olmadan yapıldı:

| Ayrı replay | Özgün seçilmiş aday | Fizik bütçesi | Kanıt |
|---|---|---|---|
| Global wave_a | `wave_a__llm_feedback__7` |1 koşu /200 kontrol /4.000 adım |[Salt okunur recovery audit](D:/projects/dance/artifacts/runs/agentic/tool_selection_compatible_smoke_v1_recovery_audit_v1/audit.json) geçti |
| Local high5_b | `high5_b__llm_feedback__6` |1 koşu /200 kontrol /4.000 adım |[Local smoke summary](D:/projects/dance/artifacts/runs/agentic/tool_selection_compatible_smoke_v2/summary.json) smoke_passed=true |

İki replay'de qpos/qvel/uygulanan PD/state_times dosyaları byte eşit; policy input/output, reference cache ve common dört array'i decompressed dtype/shape/byte eşit. Strict common-input karşılaştırması tüm200 karar için, güncel torso/FK ve bağımsız fizik kapısı denetimleriyle geçti. Preexecution pin farkı yalnız özgün executor→compatible executor değişimidir; ortak beden/model/sürüm pinleri aynıdır. Hedef metrikleri replay'de yeniden üretilmiş ayrı deney skoru olarak sunulmaz: `metrics_reused_by_trace_identity`, tam iz+referans eşitliğine bağlı özgün değerlendirmeyi taşır.

Global koşu ilk çalıştırmada fizik/preflight başarısından sonra verifier'ın Windows `\`/`/` yol ayracı karşılaştırma hatasında durmuştu. [Başarısız ilk summary](D:/projects/dance/artifacts/runs/agentic/tool_selection_compatible_smoke_v1/summary.json) korunmuştur. Yeni verifier yol anahtarlarını normalize eder, çakışmayı reddeder ve mevcut global koşuyu salt okunur denetledi; global fizik tekrarlanmadı. Local görev yeni tek-koşuluk protokolde bir kez yürütüldü. Bu iki ekstra koşu100 benchmark skoruna veya sekiz-slot arama bütçesine eklenmez; iki belirli deterministik reset/kod yolunun uyumunu gösterir, geniş sağlamlık/dağılım genellemesi değildir.

### Ayrı kalite tanıları

PD hedef step RMS/max (rad) ve gerçek eklem ivmesi finite-difference RMS/max (rad/s²), seçilmiş adaylarda:

| Görev/yöntem | PD step RMS / max | Gerçek ivme RMS / max |
|---|---|---|
| wave_a / S | 0.032763 / 0.751078 | 11.138 / 116.516 |
| wave_a / F | 0.037962 / 0.751078 | 13.744 / 305.199 |
| wave_a / O | 0.037810 / 0.751078 | 13.805 / 320.392 |
| wave_b / S | 0.032670 / 0.751078 | 11.167 / 116.516 |
| wave_b / F | 0.030834 / 0.751140 | 10.584 / 116.457 |
| wave_b / O | 0.033108 / 0.751140 | 11.528 / 116.457 |
| high5_a / S | 0.053355 / 0.893862 | 17.914 / 274.311 |
| high5_a / F | 0.053940 / 0.804536 | 18.008 / 283.757 |
| high5_a / O | 0.054406 / 0.910376 | 17.358 / 302.009 |
| high5_b / S | 0.050760 / 0.793381 | 15.072 / 166.051 |
| high5_b / F | 0.055101 / 0.868907 | 18.672 / 263.288 |
| high5_b / O | 0.056431 / 0.851202 | 19.656 / 321.264 |

Beşli hedef başarısı daha küçük PD adımı veya ivme garantisi vermedi: örneğin wave_a F gerçek ivme max305.199, sayısal seçimin116.516 rad/s² değerinden yüksek. Bu tablo bağımsız estetik/beğeni skoru veya yeniden sıralama ölçütü değildir. Committed Hermite reference eğrilerinin ayrı analitik tanısı aşağıdadır; gerçek beden ivmesiyle aynı nesne değildir.

### Committed referans eğrisinin sürekli türev tanısı

[Posthoc reference tanısı](D:/projects/dance/artifacts/runs/agentic/tool_selection_reference_diagnostics_v2/report.json) `analysis_passed=true`, errors=[]; kaynak, input ve özgün/projection inventory'leri değişmedi, yeni fizik/model çağrısı0, skor/kapı değişikliği=false. Tanı aynı array'leri taşıyan compatible projection paketlerini inceler; özgün yollar `tool_selection_development_v2/trials/{candidate}/common_reference`, projection yolları `tool_selection_development_v2_compatible_refs_v1/references/{candidate}` olarak raporda hash ile bağlıdır. `original_strict_g1_admission=false` ve `compatible_projection_required=true` burada da korunur. Analiz helper'ı yürütme freeze'inin değil, posthoc analiz anının snapshot'ıdır.

14 kol ekleminde committed Hermite q/dq eğrisinin sürekli-zaman en büyük mutlak hız ve ivmesi:

| Referans | Hız tepe rad/s | İvme tepe rad/s² |
|---|---:|---:|
| Wave global calibration |8.41072 |424.935 |
| Wave_a seçilmiş feedback |8.98365 |343.892 |
| Wave_b seçilmiş feedback |8.50706 |380.068 |
| High5 global calibration |15.54781 |892.785 |
| High5_a seçilmiş feedback |20.37609 |1369.698 |
| High5_b seçilmiş feedback |19.46233 |1621.761 |

A/B tablolarındaki aynı-source baseline tekrarları yalnız iki özgün referans paketidir; dört bağımsız baseline örneği değildir. Sayısal wave_b seçiminin sürekli hız tepesi12.99697 rad/s, knot örneklerinde11.73847 rad/s'dir: Hermite eğrisinin knotlar arasında daha büyük türev alabileceğini gösterir. **Bu tepenin motora gerçekten sorulduğu ileri sürülmez:** benchmark kararları50Hz, dört future offset'i de20ms ızgaradadır; sürekli iç tepe bu sorgulara denk gelmeyebilir. Seçilmiş high5_a/high5_b'nin dq knot örnekleri de sırasıyla yaklaşık20.3541/19.4623 rad/s değerlerine ulaşır; örnekli ve sürekli değerler ayrı raporlanır.

Bu, istenen referans hareketinin türevidir; uygulanan PD hedef hızı/ivmesi veya gerçekleşen beden hızı/ivmesi değildir. Dışarıdan10rad/s gibi bir donanım sınırı ithal edilmedi, yeni admission eşiği veya skor sonrası eleme yapılmadı. Özellikle beşli hedef başarısı, yüksek referans türevlerini azaltmış veya fiziksel kaliteyi her ölçüde geliştirmiş anlamına gelmez.

Root'un kaydettiği [v2 doğrulama çalışması](D:/projects/dance/artifacts/runs/agentic/tool_selection_verification_v2/test_run.json) exit_code=0; [test çıktısı](D:/projects/dance/artifacts/runs/agentic/tool_selection_verification_v2/stdout.txt) **114 test /8.13 s** bildirir. Bunlar runtime/schema/search/workflow/audit kontrolleridir; tamamlanmış 100 koşunun bağımsız audit'i veya ürün kalitesi kabulü değildir.

Baselines ve feedback yürütmeleri ayrı Python süreçleriyle başlatılmıştır. Kaynak/runtime import, inference ve ölçülen wall süreleri süreç başlatma, cache ve kaynak rekabeti etkileri taşır; yöntem hız sıralaması olarak yorumlanmayacaktır. Eşit öneri sayısı ve fizik slotu token, bağlam boyutu, hesaplama veya ücret eşitliği değildir. Sağlayıcı token/ücret verisi bilinmiyorsa null kalır.

## V1 karantinası ve korunmuş kanıt

[Calibration bağlam incelemesi](D:/projects/dance/design/agentic/CALIBRATION_CONTEXT_REVIEW.md), ortak calibration'ın A görevine göre signed hatalarının B canonical hedefinin yanında taşındığını saptadı. Gerçek calibration ölçümleri ve canonical hedefler doğruydu; yanlış signed hata alanı LLM bağlamını tutarsızlaştırıyordu. Etkisinin iki LLM koşuluna eşit olduğu veya model kararını etkilemediği varsayılmaz.

[V1 karantina ledger'ı](D:/projects/dance/artifacts/runs/agentic/tool_selection_development_v1_quarantine_v1/summary.json): **11 gerçek fizik koşusu =4 calibration +7 aday, 2.200 kontrol, 44.000 fizik adımı; 35 gerçek çağrı =32 off +3 feedback**. Bunlar v2 sonuçlarına eklenmez ve tamamlanmış yöntem karşılaştırması olarak sunulmaz. Bu işin toplam harcaması ayrı muhasebede **113 fizik, 22.600 kontrol, 452.000 fizik adımı ve 99 gerçek öneri turudur**: karantinadaki v1:11 fizik/35 tur; v2 benchmark:100/64; iki ayrı compatibility replay:2/0. Projection ve salt okunur recovery audit yeni fizik/öneri turu eklemedi. Mühendislik altagentları ve parent yazışmaları deneyin öneri turu sayısına dahil değildir; toplam token/ücret bilinmiyor. Eski dosyalar/hash'ler korunmuştur; düzeltme yerinde yapılmadı. Ledger kapsamı korunmuş manifest/receipt sayımıdır, tamamlanmış bağımsız sayısal audit değildir.

Eski `high5_b__local_capability__37` ve confirmation tanığı bu bağlam hatası nedeniyle ortadan kalkmaz. Ancak ayrı geliştirme araması ve bütçesidir; v2'nin sekiz slotuna aday/etiket veya başarı olarak aktarılmaz. Eski sayısal fizik tanıkları ile temiz bilgi koşulundaki yöntem karşılaştırması farklı iddialardır.

## Kalite, nedensellik ve sonraki artifactlar

İfade hedefi olarak kullanılan kol RMS oranı ve speed peak zamanı, algılanan ifade veya seyirci beğenisi değildir. PD hedef adımları, 20 ms örnekli finite-difference gerçek joint ivmesi ve hizalanmamış source-world/gerçek root displacement ayrı kalite tanılarıdır; selector'u veya önceden sabitlenmiş kapıları yeniden sıralamaz. Root koordinat ölçümünün heading correction içermemesi ayrıca sınırlamadır.

Reference tanısı tamamlandı ve yukarıda ayrı sonuçları verildi. Dört preview üretildi ve teknik/örnek-kare yerleşim QA'sı tamamlandı:

| Görev | Kayıtlı durum karşılaştırması | Hash bağlı QA receipt'i |
|---|---|---|
| wave_a |[Videoyu aç](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/wave_a/comparison.mp4) |[QA kaydı](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/wave_a/visual_qa_v1.json) |
| wave_b |[Videoyu aç](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/wave_b/comparison.mp4) |[QA kaydı](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/wave_b/visual_qa_v1.json) |
| high5_a |[Videoyu aç](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/high5_a/comparison.mp4) |[QA kaydı](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/high5_a/visual_qa_v1.json) |
| high5_b |[Videoyu aç](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/high5_b/comparison.mp4) |[QA kaydı](D:/projects/dance/artifacts/media/agentic/tool_selection_development_v2/high5_b/visual_qa_v1.json) |

Her video ffprobe ile960×720,100kare,25fps,4s H.264 ve4s AAC olarak doğrulandı. Root her görevde2s orta kareyi, seçilmiş accent'e yakın kareyi (sırasıyla73/78/55/56) ve3.96s son kareyi gördü:toplam12örnek karede tam bedenler/kalkmış eller/ayaklar görünür, başlıklar ve metrikler okunaklı. **Bu yalnız örnek-kare yerleşim kontrolüdür**; bütün karelerde estetik, insan beğenisi, temas doğruluğu veya donanım uygunluğu onayı değildir.

Panellerde `P` fizik koşusu, `M` LLM öneri turudur. Görünen baseline her kaynak için bir global calibration panelidir; iki local parity calibration da dört toplam calibration maliyetine dahildir, ayrı görünür panel değildir. Preview mevcut kayıtlı durumların `mj_forward` render'ıdır; yeni `mj_step`, politika/öneri çağrısı veya skor değişikliği üretmez. Özgün metadata'nın strict kabul başarısızlığı, compatible projection audit'i ve iki sonradan yapılan strict replay yukarıdaki kapsamlarıyla aynen geçerlidir.

Feedback/off farkı yalnız bu iş akışındaki, bu dört ilişkili görev ve sekiz-slot bütçesindeki gözlenen fark olarak raporlanabilir. Feedback bağlamında ek ölçüm bilgisi ve buna bağlı token/bağlam artışı bulunur; salt token miktarından bağımsız model mekanizması veya genel nedensel üstünlük çıkarılmaz. İnsan/VLM kör değerlendirmesi yapılmadığından estetik veya insan beğenisi sonucu yoktur. Training admission=false, kalite=unassessed; yeni ağırlık öğrenimi, canlı recovery, genel beden aktarımı ve ürün dağıtım kabulü bu çalışmanın sonucu değildir.
