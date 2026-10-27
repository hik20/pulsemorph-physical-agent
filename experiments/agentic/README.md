# Agentic fiziksel düzenleme pilotu

Güncel tamamlanan çalışma: [iki araçlı geliştirme karşılaştırması v2](TOOL_SELECTION_RESULTS.md). 100 gerçek fizik koşusu ve64 yeni LLM öneri turunda, dört kol hedefi ve bir müzik vurgusu zamanını birlikte sağlayan görev sayısı feedback4/4, feedback-off1/4, sayısal arama0/4. Hedef bağımsız kalibrasyon kullanıldı. Özgün paketlerin strict metadata uyumsuzluğu, array'leri değiştirmeyen100uyumlu dışa aktarım ve bağımsız girdi eşleşmesiyle açıkça ayrıldı; düzeltilmiş yürütücüde iki ek fizik replay'i de geçti. Dört video ve referans türev tanısı hazır. Sonuç iki tekrar kullanılan geliştirme kaynağıyla sınırlı; high5_b dar tolerans marjlı, algısal kalite/sağlamlık kabulü yok.

Sonraki incelemede eski çok özellikli/genelleme LLM deneylerinin B görevlerine, kalibrasyonun A hedefine göre hata alanlarının da gösterildiği bulundu. Gerçek aday ölçümleri ve skorları korunur; eski LLM başarı oranları temiz bir geri bildirim etkisi sayılmıyor. [Ayrıntı ve yeni sürüm gerekçesi](../../design/agentic/CALIBRATION_CONTEXT_REVIEW.md). İki araçlı development v1 bu kusurla karantinaya alındı; hedef bağımsız kalibrasyon görünümüyle v2 tamamlandı.

Önceki tamamlanan çalışma: [high5_b araç kapasitesi araştırması](HIGH5_CAPABILITY_RESULTS.md) — 105 yeni gerçek fizik koşusu; global araçta64aday çözüm bulamadı, ayrı yerel zaman aracı38.adayda beş hedefi sağladı ve tek fresh-reset tekrarı geçti. İki bağımsız denetim ve sesli videolar korunur. Daha önceki [kaynak dosyası karşılaştırmasının](GENERALIZATION_RESULTS.md) 3/4–1/4–0/4 kayıtlı sonucu yukarıdaki bağlam kusuruyla birlikte okunur. [Tek kaynakta çok özellikli deney](MULTIFEATURE_RESULTS.md) ve aşağıdaki tek parametreli pilot tarihçe olarak korunur.

26 Ekim 2026. Gerçek `gpt-6.1-sol` önerisi → referans düzenleme → pretrained ONNX → PD → MuJoCo → sayısal gözlem → yeni öneri döngüsü çalıştırıldı. Model önerileri Codex collaboration üzerinden alındı; bağımsız sağlayıcı API entegrasyonu veya dağıtılabilir otonom servis değildir. Yeni ağırlık eğitilmedi.

## Sonuç

Tek kaynak dansın dört saniyelik bölümünde, 1–3 saniye arasında kol referansı sin² zarfıyla küçültüldü. Ölçüm, fiziksel eklem açılarının ortak başlangıç kol duruşundan 1,5–2,5 saniye arasındaki RMS uzaklığıdır. Hedef, değişmemiş take'in **0,70** katı; tolerans **±0,05**. Bu metrik algılanan genişlik veya insan beğenisi değildir.

Her yöntem üç aday ve üç gerçek fizik koşusu kullandı. Ortak değişmemiş kalibrasyonla toplam **10 koşu, 2.000 politika çağrısı, 40.000 fizik adımı**. Önceki kısa yürütücü smoke deneyi ayrıca 2 politika/40 fizik adımıdır; karşılaştırmaya ek aday olarak alınmadı.

| Seçilen sonuç | Gerçekleşen oran | Hedefe mutlak oran hatası | Eklem izleme RMSE | Yerel fizik kapısı | Hedef toleransı |
|---|---:|---:|---:|---|---|
| Basit secant araması | 0,700234 | 0,000234 | 0,204841 rad | Geçti | Geçti |
| LLM + fizik geri bildirimi | 0,699666 | 0,000334 | 0,204699 rad | Geçti | Geçti |
| LLM, revizyona geri bildirim kapalı | 0,684104 | 0,015896 | 0,204132 rad | Geçti | Geçti |
| Değişmemiş kalibrasyon | 1,000000 | 0,300000 | 0,227789 rad | Geçti | Hedef düzenlemesi yok |

Geri bildirimli model, 0,5 → 0,525 → 0,52395 genlik önerdi; gerçekleşen oranlar 0,68410 → 0,70069 → 0,69967 oldu. Basit arama biraz daha düşük son seçim hatası verdi. Bu pilot **LLM'nin basit aramaya üstünlüğünü göstermedi**. Geri bildirim hassasiyeti artırdı; bütün yöntemlerin seçilen sonucu zaten önceden belirlenen toleransı geçti. Aynı ilk 0,5 adayı üç yöntem için zorunluydu ve gerçek qpos/qvel/PD hedef izleri bit düzeyinde aynı çıktı.

Geri bildirim kapalı modelin 0,5 / 0,67 / 0,73 önerileri hiçbir aday koşusu başlamadan taahhüt edildi. Son iki adayın oranları 0,79647 ve 0,83122 ile talimat toleransını geçemedi. Bütün yöntemler için aynı bağımsız son seçici kullanıldı: yerel fizik kapısından geçenler arasında en düşük oran hatası. “Geri bildirim kapalı” yalnız teklif revizyonunu anlatır; son seçici fizik sonuçlarına erişir.

Yerel kapı önceden sabitlendi: bütünlük denetimi, en düşük kök yüksekliği ≥0,60 m ve ortalama zaman hizalı eklem RMSE ≤0,35 rad. Bu kapı kapsamlı denge, temas kalitesi, donanım güvenliği veya canlı runtime kabulü değildir. Seçilen üç adayın en düşük yüksekliği 0,73654 m.

## Ayrı ölçümler ve yan etkiler

- Bütün adaylarda alt beden **referansı** aynı kaldı. Gerçekleşen alt beden değişti: seçilen sonuçlarda baseline'a karşı yaklaşık 0,048–0,049 rad RMSE. Fiziksel etkileşim nedeniyle “ayak hareketi kilitlendi” iddiası yok.
- Referans düzenlemesi 1 saniyede başlasa da tracker +160 ms öngörü tüketir. Gerçek durum farkı **0,88 saniyede** başladı. Bu tam take düzenlemesidir; canlı geçmişi değiştirme veya 1 saniyeye kadar fiziksel kilit garantisi değildir.
- Sabit authored sesin bilinen dosya beat haritasıyla hız tepesi vekili değerlendirildi. En yakın vuruş hatası baseline'da 123,3 ms; seçilen basit/LLM/geri bildirim kapalı sonuçlarda 108/104/108 ms. Ancak eşleşme **6/8'den 5/8'e** düştü. Daha düşük hata genel müzikal iyileşme diye sunulamaz. Ses yeni koreografi üretmiyor; kaynak zamanlaması bu pilotta sabit.
- Üç koşunun toplam yürütücü duvar süresi basit aramada 44,67 s, geri bildirimli LLM'de 44,94 s, geri bildirim kapalıda 27,66 s. Aynı süreçte sıcak başlangıç ve farklı host yükü nedeniyle bunlar yöntem hız sıralaması değildir. Seçilen adaylarda ONNX p50/p95/p99: basit 1,78/2,45/3,31 ms; LLM 2,00/3,39/5,09 ms; kapalı 1,90/2,89/3,37 ms. Uçtan uca canlılık garantisi yok.
- Model öneri çağrıları 0 / 3 / 1, aday sayısı her kolda 3. LLM token sayısı, sağlayıcı ücret makbuzu ve saf model gecikmesi erişilemiyor; **null**, sıfır değil. Araştırma/uygulama sırasında kullanılan agent çalışması bu deney çağrı sayısına dahil değildir. Basit aramanın model token maliyeti 0; donanım enerji maliyeti ölçülmedi.
- İnsan tercihi, serbest dil ayrıştırma doğruluğu, video/VLM katkısı, yeni beceri öğrenimi ve görülmemiş kaynaklara genelleme ölçülmedi.

## Kod ve sözleşme

- [Referans aracı](../../src/pulsemorph/agentic/reference.py): analitik türevle tutarlı q/dq, torso xyzw, kaynak zamanında Hermite/SLERP, sınırlı genlik/hız. Retiming test edildi; bu fizik pilotunda hız 1.
- [Yürütücü](execute_trial.py): sabit yerel source/model, tam ONNX girdileri/çıktıları, PD hedefleri, gerçek durumlar, her fizik adımı için sonluluk/saat/uyarı denetimi, değişmez çıktı.
- [İş akışı](workflow.py) ve [pilot sürücüsü](pilot.py): teklif bütçesi, gözlem–yanıt bağlantısı, taahhüt, fizik koşusu, geri bildirim, ortak seçim. LLM önerileri yalnız izinli sayısal parametreye dönüşür; model metni kod olarak çalıştırılmaz.
- [Bağımsız değerlendirici](evaluate_trial.py): giriş/çıktı byte hashleri, gerçek ONNX joint girdileri, torso heading hizalaması, uygulanan PD hedefi, başlangıç/ortam/profil eşitliği ve ölçümler.
- [Ön kayıt ve yöntem dayanakları](../../design/agentic/FIRST_EXPERIMENT.md).

İlk fizik koşuları **yerel taslak** adaptörle yürüdü. Ana sohbetin ortak `pulsemorph.motion` kodu bu sırada kullanılabilir hâle gelince [export_common.py](export_common.py) ile baseline ve seçilen üç aday ayrıca gerçek `pulsemorph.motion_reference.v1` paketlerine aktarıldı. [Ortak export/parity raporu](../../artifacts/runs/agentic/compact_arms_v1_common_export_v2/report.json): dört save/load doğrulaması ve **800 kararın** kaydedilmiş gerçek ONNX reference girdisiyle karşılaştırması geçti. Joint açı ve hız hatası0; en büyük torso bileşen farkı7,45e-9, tolerans2e-6. Ortak kodun hash/snapshot'ı saklandı ve export sırasında değişmedi.

Her ortak paket0..4,14s arasında208 örnek içeriyor; her pakette793 çakışan cache örneği birebir eşleşti. Gerçek yürütme4s; son0,14s lookahead. Bu sonuç **reference-input parity** ve ortak veri doğrulamasıdır; ortak adaptörle yeni fizik replay'i veya örnekleme ızgarası dışında türev eşdeğerliği değildir. İlk export'ta null observation_id reddi v1 hata paketinde korundu; geçerli kimlikle v2 üretildi. Eski fizik kanıtları değiştirilmedi.

Full-body taşıyıcı kaynak interpolasyonunu tutar ve düzenlenmiş kolun FK sonucu değildir; ortak pakete yalnız gerçek tüketilen eklem/torso hedefleri alınır. Heading hizalaması bir kez uygulanır. Body/controller hashleri, source-world frame, başlangıç durum fingerprint'i, offline gözlem erişimi ve kaynak soyu ortak metadata'da var. Bu export canlı runtime kabulü veya recovery garantisi getirmez.

`trial_records/` altında öneri, gözlem, model yanıtı, istenen referans, uygulanan PD hedefi, gerçek hareket ve bağımsız değerlendirme ayrı bağlantılardır. Bunlar ortak TrialRecord uygulaması değildir; öğrencinin veri alımı için aday pakettir. Kaynak kökeni upstream generated motion olarak kalır; hazır tracker becerisi yeni öğrenme sayılmaz.

## Kanıt ve yeniden çalıştırma

Ana paket: [compact_arms_v1](../../artifacts/runs/agentic/compact_arms_v1/comparison.json). [Protokol](../../artifacts/runs/agentic/compact_arms_v1/protocol.json) aday sonuçlarından önce yazıldı. [Son bağımsız denetim](../../artifacts/runs/agentic/compact_arms_v1/final_independent_audit.json) bütün 10 koşunun ölçümlerini aynı bit değerleriyle yeniden hesapladı. Yerel kaynak snapshot'ları, metadata, gerçek arrayler ve ilk/son frozen hash kontrolü saklandı. Korunan learned-rhythm v2 dosyaları değişmedi.

[Dört panel video](../../artifacts/media/agentic/compact_arms_v2/comparison.mp4), kayıtlı **gerçek fizik qpos** değerlerinin `mj_forward` ile render'ıdır; yeni fizik replay koşusu değildir. 100 kare/25 FPS/4 saniye FFprobe ile doğrulandı; orta kare görsel olarak incelendi. İlk render'ın zemin adı hatası [ayrı başarısız dizinde](../../artifacts/media/agentic/compact_arms_v1/render_failure.json) korundu. Fizik deneyleri etkilenmedi.

35 test + 9 alt kontrol geçti: 29 yeni agentic testi ve 6 mevcut causal reference regresyonu. [Komut ve stdout sonu](../../artifacts/runs/agentic/compact_arms_v1/verification/targeted_tests.json). Global pytest eklenti ortamında eksik `sniffio` nedeniyle otomatik eklenti yükleme kapatıldı; paket kurulmadı.

```powershell
$env:PYTHONPATH='src;.;experiments/agentic'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv-motor/Scripts/python -m pytest tests/test_agentic_reference.py tests/test_agentic_evaluation.py tests/test_agentic_workflow.py tests/test_causal_reference.py -q

# Her yeniden koşuda YENI_ID; mevcut kanıt üzerine yazılmaz.
.venv-motor/Scripts/python experiments/agentic/workflow.py --initialize artifacts/runs/agentic/YENI_ID
# Gözlem dosyalarından gerçek model yanıtları sağlanır; feedback-off üç yanıtı baştan taahhüt eder.
# Model yanıtı replay edilirse yeni model çağrısı sayılmaz.
.venv-motor/Scripts/python experiments/agentic/execute_trial.py --proposal artifacts/runs/agentic/YENI_ID/proposals/calibration_0.json --output artifacts/runs/agentic/YENI_ID/trials/calibration_0
.venv-motor/Scripts/python experiments/agentic/pilot.py --pilot artifacts/runs/agentic/YENI_ID --action evaluate-calibration
.venv-motor/Scripts/python experiments/agentic/pilot.py --pilot artifacts/runs/agentic/YENI_ID --action commit-off
.venv-motor/Scripts/python experiments/agentic/pilot.py --pilot artifacts/runs/agentic/YENI_ID --action run --method llm_feedback --index 0
# Sonraki feedback yanıtı kendi yeni observation dosyasından üretilir; her yöntem index0..2.
.venv-motor/Scripts/python experiments/agentic/pilot.py --pilot artifacts/runs/agentic/YENI_ID --action summarize

# Mevcut gerçek koşuyu yeniden simüle etmeden ortak paket ve girdi eşleşmesi:
.venv-motor/Scripts/python experiments/agentic/export_common.py --pilot artifacts/runs/agentic/compact_arms_v1 --output artifacts/runs/agentic/YENI_EXPORT_ID
```

Sonraki anlamlı deney: ortak MotionReference adaptörüyle birden çok ifade parametresi ve çatışan talimat; görülememiş kaynak/başlangıçlar; güçlü sayısal arama; bağımsız görsel/insan değerlendirmesi. LLM'nin yalnız parametre optimizasyonuna eklenmesi zorunlu mimari kararı değildir.
