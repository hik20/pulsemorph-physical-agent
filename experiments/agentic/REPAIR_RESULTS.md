# Başlangıç hassasiyeti onarım deneyi — sonuç

27 Ekim 2026. Wave_a'nın bilinen paneldeki tek hedef kaybı iki LLM kolunda giderildi; high5_b tam onarılmadı. Bu, iki seçilmiş geliştirme görevinin ölçümüdür; genel LLM üstünlüğü, feedback'in nedensel yararı veya insan beğenisi sonucu değildir.

[Makine okunur rapor](../../artifacts/runs/agentic/repair_development_v1_report_v1/report.json), [grafikler ve tablo](../../artifacts/runs/agentic/repair_development_v1_report_v1/report.md), [bağımsız yorum](../../design/agentic/REPAIR_RESULTS_REVIEW.md) ve [sonraki deney tasarımı](../../design/agentic/REPAIR_NEXT.md) ayrı korunur. Protokol SHA256: `7fbd7e565330183cd4c3d08f0067431e4aad41dbc3d2ab96b95ce6d07a0e4ec9`.

| Görev | Yöntem | Nominal /1 | Nonzero /12 | Tam /13 | Tam en kötü normalize hata | Marj: 1−hata |
|---|---|---:|---:|---:|---:|---:|
| wave_a | Eski incumbent | 1 | 11 | 12 | 5.932200 | −4.932200 |
| wave_a | Sayısal minimax | 1 | 11 | 12 | 1.065195 | −.065195 |
| wave_a | LLM + feedback | 1 | 12 | 13 | .932200 | .067800 |
| wave_a | LLM feedback off | 1 | 12 | 13 | .932200 | .067800 |
| high5_b | Eski incumbent | 1 | 4 | 5 | 1.456902 | −.456902 |
| high5_b | Sayısal minimax | 0 | 0 | 0 | 1.396499 | −.396499 |
| high5_b | LLM + feedback | 0 | 0 | 0 | 1.726538 | −.726538 |
| high5_b | LLM feedback off | 1 | 7 | 8 | 1.831065 | −.831065 |

Seçim dört geliştirme koşulunda önce bütün fizik kapıları, sonra en kötü normalize hata, kareler toplamı ve aday kimliğidir. High5_b geliştirme en kötü hataları sayısal 1.348073, feedback 1.344096, off 1.414959'dur. Sayısal çıktı dört geliştirme koşulunun tamamında hedefi kaçırır; daha düşük minimax hata daha fazla geçiş demek değildir. Altı çıktı dokuz ek koşuldan önce dondurulmuş; sonradan yeniden seçim yapılmamıştır. Eski 24 nominal aday + iki identity araç kaydı + incumbent'ın 13 koşulu bütün kollara eşit verilmiştir; tekrarlanan nominal trace bağımsız ek kanıt sayılmaz. Sayısal karşılaştırma iki aracı adaptif seçen minimax/trust-region aramasıdır; mümkün bütün optimizer'ların en güçlüsü olduğu iddia edilmez.

Wave sayısal kaybı `root_vy_plus` altında left-B genişlik hatasıdır (+.085216; tolerans .08). Eski incumbent `combined_minus` altında vurgu peak'ini 2.4s seçer. İki LLM çıktısında bütün vurgu peak'leri 2.9s, hedef 2.993219955s ve zaman kabul payı **6.780045ms**'dir; 13/13 geçiş bu dar zaman payının genişlediğini göstermez. Posthoc top-two peak yüksekliği farkının en küçüğü feedback'te **.001316847rad/s**, off'ta **.024471168rad/s**; ikisi de `combined_minus` altındadır. Bunlar yeniden sıralama ölçütü veya olasılıksal dayanıklılık kanıtı değildir. Sayısalda minimum fark .032072083rad/s (`root_vy_plus`), eski incumbent'ta .000329933rad/s (`root_vy_plus`). High5'te çoğu koşulda yalnız tek event olduğundan top-two farkı NA'dır; bütün NA gerekçeleri raporda saklanır.

High5 sayısal kolunda 13 koşulun tamamında vurgu 2.16s ile hedefi kaçırır; ek left-B 9/13 ve right-B 10/13 kaybı vardır. Tam en kötü hata `joint_position_plus` right-B'dedir. Feedback'te bütün koşullarda en az bir genişlik hedefi kaçar: right-A 10/13, left-B 3/13, right-B 11/13; en kötü durum `root_vy_minus` right-B'dir. Off'un beş kaybı `joint_position_plus` (right-B), `joint_position_minus` (left-B), `root_vy_plus` (left-B), `combined_plus` (right-B), `combined_minus` (left-B); en kötü `joint_position_plus` right-B'dir. Her adayın bütün başarısız koşulları, signed feature hataları ve kaynak bağlantıları rapor JSON'unda korunur.

**250 yeni fizik** = 4 runtime parity gate + 192 aday (2×3×8×4) + 54 kalan confirmation; 50.000 kontrol ve 1.000.000 fizik adımı. **32 yeni gerçek model öneri turu**, mühendislik çağrıları hariç. Token ve ücret ölçülmedi. `gpt-6.1-sol` çağrılarda istenen model alias'ıdır; değişmez sağlayıcı ağırlık revizyonu/receipt yoktur (null). İki kolda aynı belirtilmemiş reasoning-effort ayarı kullanıldı, override yoktur; motor ONNX/checkpoint hash'leri ayrıca sabittir.

Dört geliştirme satırı tam panelde yeniden kullanılır; incumbent seçilseydi dört eski prior satırı kullanılacaktı. Dokuz ek koşul seçime girmemiş olsa da 13 koşulun tamamı bilinen geliştirme koşuludur, temiz holdout değildir. 246 değerlendirilen koşunun tamamı ve dört kontrol koşusu tanımlı fizik kapılarını geçti. PD hedef adımları ve 20ms örnekli finite-difference joint ivmesi ayrı tanıdır; seçimde kullanılmaz, ara-adım ivmesini veya insan kalite değerlendirmesini ölçmez. Canlı recovery, ağırlık öğrenimi, yeni beden/kaynak genellemesi kanıtı yoktur.

[İlk doğrulama](../../artifacts/runs/agentic/repair_verification_v1/test_run.json) 30 test + dört alt kontrol içerir; mock tam panel testi gerçek fizik yerine geçmez. [Auditor v2 doğrulaması](../../artifacts/runs/agentic/repair_audit_v2_verification_v1) dört ayrı testtir. [İlk tam audit](../../artifacts/runs/agentic/repair_development_v1_audit_v1/audit.json) göreli/mutlak gate seed yolu karşılaştırmasında başarısız olmuş ve korunmuştur. Bu auditor kusurunu düzelten [v2 tam bağımsız audit](../../artifacts/runs/agentic/repair_development_v1_audit_v2/audit.json) geçti: 250 trial, altı stream; 276 ham artifact doğrulaması ve 738 doğrulanmış cache kullanımı. Yeni fizik, model çağrısı veya skor/seçim değişikliği yapılmadı.

**Açık metadata istisnası:** raw execution proposal'ları seed'den eski `trial_index`, `brief_reason`, `created_at_utc`, `search_decision` taşıdı. `raw_metadata_semantically_correct=false`, `raw_execution_metadata_complete=false`, `legacy_metadata_disclosure_required=true`; kabul kapsamı `numerical_active_lineage_with_legacy_metadata_exception`'dır. [Ayrı 250-satır inceleme](../../artifacts/runs/agentic/repair_legacy_fields_review_v1/report.json) ve [annotations](../../artifacts/runs/agentic/repair_legacy_fields_review_v1/annotations.json) güncel kimlik/araç/parametre/response ve sayısal yürütme girdilerine etkisini sınırlar; temiz bütün-metadata provenance iddiası sağlamaz. Güncel karar üst-seviye `proposals/<repair_candidate_id>.json` ve hash bağlı response/observation'dadır. Ham proposal'lar yerinde düzeltilmedi; metrikler değişmedi ve yeniden sıralanmadı.

![Dört koşul arama eğrileri](../../artifacts/runs/agentic/repair_development_v1_report_v1/development_best_so_far.png)

![Tam panel hata ve geçişleri](../../artifacts/runs/agentic/repair_development_v1_report_v1/full13_heatmaps.png)

## Kayıtlı fizik görüntüleri

[Wave_a / combined_minus](../../artifacts/media/agentic/repair_development_v1/wave_a/comparison.mp4) ve [high5_b / joint_position_minus](../../artifacts/media/agentic/repair_development_v1/high5_b/comparison.mp4), eski çıktı ile üç seçilmiş yöntemi aynı başlangıç koşulunda gösterir. Her video960×720/100kare/25fps/4s ve aynı görevin4s sesini içerir. Bunlar bütün13 koşulun temsili özeti değil, açıkça adlandırılmış iki tanı örneğidir. Kaydedilmiş qpos üzerinde yalnız mj_forward ile render edildi; yeni fizik veya model çağrısı yoktur. Her videonun orta/vurguya yakın/son kareleri incelendi: beden ve metin görünürlüğü geçti, estetik beğeni ölçülmedi. Manifestler ve görsel QA kayıtları her videoyla aynı klasördedir.
