# Müzik ve birleşik talimatla çok özellikli düzenleme

**26 Ekim 2026 ek kapsam düzeltmesi:** B görevinin 9 LLM observation'ında ortak kalibrasyondan A hedefine ait hata vektörü ve `utility_l2` taşındığı sonradan görüldü. Ham kalibrasyon özellikleri, canonical hedefler ve 48 adayın kendi hedefine göre skor aritmetiği doğrudur; kayıtlı fiziksel başarılar korunur. Aşağıdaki yöntem sayıları tutarsız başlangıç bağlamı altındaki betimleyici sonuçlardır; temiz bir geri bildirim etkisi veya LLM üstünlüğü kanıtı sayılmaz. [Kapsam incelemesi](../../design/agentic/CALIBRATION_CONTEXT_REVIEW.md). Özgün deney dosyaları değiştirilmedi.

26 Ekim 2026. Deney ve bağımsız paket denetimi tamamlandı. Bu çalışma yeni ağırlık eğitimi değil, gerçek LLM ve hareket araçlarıyla öneri–fizik–gözlem–düzeltme deneyidir.

## Sonuç

Geri bildirimli `gpt-6.1-sol`, iki görevde de **dört kol hedefini ve müzik vurgusunu aynı anda** önceden belirlenen toleranslara getirdi. Her yöntem görev başına sekiz gerçek fizik koşusu aldı. Yerel Jacobian/Broyden araması ve geri bildirimsiz LLM birer görevde bütün hedefleri tuttu.

| Görev | Yöntem | İlk bölüm sol / sağ | İkinci bölüm sol / sağ | Baskın vurgu hatası | Beş hedef birlikte |
|---|---|---:|---:|---:|---|
| A | Jacobian araması | 0,653 / 1,115 | 1,062 / 0,749 | 214,3 ms | Geçmedi |
| A | **LLM + fizik geri bildirimi** | **0,701 / 1,150** | **1,150 / 0,701** | **5,7 ms** | **Geçti** |
| A | LLM, revizyona geri bildirim kapalı | 0,789 / 1,165 | 1,147 / 0,736 | 114,3 ms | Geçmedi |
| B | Jacobian araması | 1,189 / 0,726 | 0,721 / 1,168 | 53,2 ms | Geçti |
| B | **LLM + fizik geri bildirimi** | **1,143 / 0,730** | **0,710 / 1,155** | **46,8 ms** | **Geçti** |
| B | LLM, revizyona geri bildirim kapalı | 1,096 / 0,726 | 0,661 / 1,193 | 6,8 ms | Geçti |

A hedefleri ilk bölümde0,70/1,15; ikinci bölümde1,15/0,70. B hedefleri bunun tersi. Oranlar, aynı kolun değişmemiş kalibrasyonda başlangıç eklem duruşundan RMS uzaklığına göredir; algılanan kol genişliği veya insan beğenisi değildir. Kol toleransı±0,08; vurgu toleransı±100ms. Tablodaki sol/sağ robotun kendi beden taraflarıdır.

İki görevde de LLM üçüncü fizik adayında ilk tam başarıya ulaştı. B görevinde geri bildirimsiz yöntem ikinci adayda, Jacobian araması sekizinci adayda geçti. Dolayısıyla **LLM her koşulda daha az deneme kullanmadı**. Bu ilk-geçiş sayıları geçmiş kayıtların açıklayıcı özeti; birincil sonuç sekiz adaydan ortak seçicinin seçtiği sonuçtur.

Görev A'da katkı somut: ilk LLM düzenlemesi dört kol hedefini tutarken baskın hareket tepesi2,30s'ye kaydı; müzik hedefi2,5143s idi. Fizik geri bildirimiyle zaman kaydırması ve kol oranları birlikte düzeltildi, seçilen tepe2,52s oldu. Genlik değişikliği hangi tepenin baskın olduğunu da değiştirdi; yerel Jacobian'ın düzgün tek tepeli yanıt varsayımı bu durumda yetersiz kaldı. Bu açıklama gözlenen deneme izlerinden çıkarımdır; bütün optimizasyon yöntemlerine üstünlük kanıtı değildir.

Görev B'de kapalı yöntem müzik zamanlamasında seçilen LLM'den daha iyi, kol hedeflerinde ise daha yüksek hata verdi. Ortak seçici önce fizik kapısını uygular, sonra toleransa bölünmüş **en kötü boyut hatasını** küçültür; kare hata toplamı ve aday kimliği eşitlikleri çözer. Bu görev seçme kuralıdır, genel dans kalite puanı değildir. Bütün boyutlar ayrı saklandı.

## Uygulanan görev, müzik ve araç

Hazırlanmış Türkçe talimatlar iki bölüm arasında sol/sağ rollerini değiştiriyor: bir kol dar, öteki geniş; ikinci bölümde tersi. İkinci yarının baskın hareket vurgusu müziğin en güçlü bas olayına taşınıyor. Başlangıca ve bitişe yumuşak dönüş isteniyor. Sözlük açık: dar0,70; geniş1,15; normal1,00.

İki özgün dört saniyelik ses dosyası kick/snare/akor/hi-hat içerir; kökenleri CC0 prosedürel sentezdir. Authored güçlü bas başlangıçları2,50s ve3,00s; gerçek PCM üzerinde140Hz low-pass ve20ms RMS ölçümünün bulduğu hedefler **2,514308s ve3,013175s**. Authored doğrular analiz sonucu yerine kullanılmadı. Algı denetiminde hata14,3ms ve13,2ms; bu iki sentetik örnek genel müzik algısı kanıtı değildir.

LLM ham PCM dinlemedi; gerçekten dalga biçiminden çıkarılan özellik JSON'unu gördü. Model yalnız Türkçe talimat, sözlük ve müzik gözlemiyle yapılan ayrı semantik çağrıda iki hedef dizisini ve doğru bas olayını eşledi. Ardından **sayısal aşamada bütün yöntemlere aynı canonical hedef verildi**. Semantik başarıyla optimizasyon başarısı aynı iddia değildir; açık sözlüklü iki talimat serbest dil genellemesini kanıtlamaz.

Araç beş parametre kullanır: iki bölüm × sol/sağ kol genliği ve zaman kaydırması. Genlik aralığı[0,35;1,60], zaman kaydırması[-0,30;+0,30]s. Kaynak saati `t + shift*sin²(pi*t/4)`; her bölümde genlik sin² zarfıyla uygulanır. q/dq analitik zincir kuralıyla tutarlı; kaynak saati monoton. Genlik pencereleri0,4–2,0s ve2,0–3,6s, RMS ölçümleri0,8–1,6s ve2,4–3,2s. Retiming alt bedeni de etkiler; ayakların veya alt bedenin sabit kaldığı iddia edilmez.

Fiziksel vurgu, gerçekleşen29 eklem hızının RMS eğrisinde100ms yumuşatma,0,15rad/s prominence ve0,35s tepe ayrımıyla bulunan[1,8;3,2]s aralığındaki en yüksek tepedir. Bütün tepe zamanları, tepe büyüklükleri ve olay sayıları saklandı. Eksik tepe uydurulmuş sıfır zamanla doldurulmaz. Bu kinematik vekil, algısal vurgu veya bütün beat'lere uyum ölçümü değildir.

## Karşılaştırma ve fizik kanıtı

- **49 gerçek koşu, 9.800 politika çağrısı, 196.000 fizik adımı**: iki görev × üç yöntem × sekiz aday + bir ortak kalibrasyon. Bu deneyde ek fizik smoke veya bütçe dışı kalite araması yapılmadı.
- İlk aday bütün akışlarda[1,1,1,1,0]. Kalibrasyon verisi ortak; her aday gerçekten yeniden yürütüldü. Geri bildirimsiz16 öneri, herhangi bir aday koşusu başlamadan taahhüt edildi.
- Jacobian baseline: ortak anchor, beş bağımsız sonlu fark probu, iki bounded damped least-squares düzeltme; son adım gerçek sonuçla Broyden güncellemesi. Geçerli fizik gözlemi olmayan aday düzeltme merkezi seçilmez. Bu güçlü bir yerel karşılaştırmadır; küresel/Bayesian/CMA-ES optimizasyonlarının tamamını temsil etmez.
- **49 koşunun tamamı yerel fizik ve bütünlük kapısını geçti.** Kapı min root≥0,60m, mean joint RMSE≤0,35rad; ayrıca hash, clock, finite state ve MuJoCo uyarıları. Seçilen sonuçlarda ortalama eklem RMSE0,2201–0,2389rad; minimum kök yüksekliği0,7274–0,7340m. Kapsamlı denge, ayak kayması, temas estetiği veya donanım güvenliği kabulü değildir.
- Ortak `MotionReference` her koşuda kaydedilip yeniden açıldı ve **gerçek kontrol yolunda örneklendi**. Sadece sonradan export yapılmadı. Hedef, PD hedefi ve gerçekleşen qpos/qvel ayrı.
- Her kontrol sınırında `mj_forward` yapıldı: koşu başına200 FK refresh. Gerçek policy torso girdileri, kaydedilmiş qpos'tan bağımsız yeniden hesaplanan FK ile karşılaştırılıyor. Eski pilotun olası1ms derived-state gecikmesi yeni gözlemle karıştırılmaz; eski kanıtlar değiştirilmedi.
- Bütün denemeler **fresh reset**. Önceki fizik snapshot'ından dallanma veya canlı geçmişi değiştirme yok. Başlangıç referans eklem fingerprint'i ve gerçek başlangıç qpos/qvel hashleri ayrı.

## Maliyet, zaman ve iddia sınırları

Her görevde model teklif çağrıları Jacobian/LLM/kapalı için0/8/1. Ayrıca iki görevi beraber ayrıştıran bir semantik model çağrısı ve deney dışı mühendislik agent çalışması var. Token ve sağlayıcı fatura makbuzu erişilemiyor; null, sıfır değil. Geri bildirimli ve kapalı koşulların model çağrı/token bütçeleri eşlenmedi: bu karşılaştırma yalnız geri bildirim bilgisinin saf nedensel etkisini izole etmez. Fizik/aday bütçeleri eşittir.

Sekiz koşunun yürütücü duvar süreleri A'da107,3/177,9/102,3s; B'de140,7/157,1/98,6s. Yürütmeler kısmen paralel, süreçler bazen sıcak bazen soğuk başladı. Bu süreler yöntem hız sıralaması değildir; model düşünme ve orchestration gecikmesi ayrıca izole edilemedi. Seçilen koşuların ONNX p50/p95/p99 değerleri JSON'da var; uçtan uca gerçek zaman iddiası yok. CPU'da mevcut motor ortamı kullanıldı; GPU/eğitim/indirme/harici API işi başlatılmadı.

Bir ilk model dosyası tek parametre vektörünü düz listeyle yazdı. İlk çağrı fizik başlamadan reddedildi. Özgün cevap korunarak yalnız dış liste eklendi; sayılar değişmedi, ek model çağrısı/aday/fizik koşusu oluşmadı. Normalizasyon teklifte açık etiketlendi ve test edildi.

**Tek motion source_group** var: iki farklı müzik ve talimat, aynı kaynak dansı bağımsız hareket testi yapmaz. Bütün kayıtlar `unassigned`, kalite `unassessed`, `training_admission=false`. Başarılı trial otomatik eğitim etiketi değildir. Yeni ağırlık, video becerisi, insan beğenisi, farklı dansçı/beden/kaynak genellemesi veya hazır ürün üstünlüğü iddia edilmez.

Sonraki ana hat [motor kabiliyeti bulgusu](../../design/learning/READINESS_REPORT.md) bu sınırı pekiştiriyor: walk rate1.15 koşulunda düşük sayılabilecek joint RMSE'ye rağmen source root travel farkı yaklaşık0,78m. Mevcut checkpoint bağımsız root rota/temas hedefi tüketmez; bizim kol/zamanlama sonuçlarımız rota, alan sınırı, geçiş veya recovery başarısına genişletilemez. Bu ana hat kanıtı burada yeniden çalıştırılmadı ve agentic koşu sayısına eklenmedi. [Veri kullanım kapsamı](../../design/learning/DATA_READINESS.md) raw veri, model ağırlığı, kod ve üretilmiş hareket için ayrı tutulur; bu rapor yeni eğitim/dağıtım uygunluğu kararı vermez.

Sonraki [geçiş pilotu](../motion_transition/RESULTS.md), referans C1 sürekliliğinin daha küçük PD hedef adımını garanti etmediğini gösterdi. Bizim talimatlarımızdaki “başlangıç ve bitişe yumuşak dön” ifadesi referans zarfıyla temsil edildi; bağımsız motor komutu, gerçekleşen ivme/torque veya algılanan akıcılık kabulü önkayıtta yoktu. Dolayısıyla “beş hedef birlikte geçti” ifadesi bu beş sayısal boyutla sınırlıdır; bütün doğal dil niteliklerinin veya komut yumuşaklığının doğrulandığı anlamına gelmez. Eski başarı/başarısızlık etiketleri ve seçimler değiştirilmedi.

Mevcut kayıtların [ek PD hedef adımı denetimi](../../artifacts/runs/agentic/multifeature_pd_steps_v1/report.json) tamamlandı: 49 koşu × 199 komut geçişi, 4.852 doğrulanmış girdi hash'i. Seçili altılıda 20 ms aralıklı hedeflerin koşu içi maksimum bileşen farkları 0,931214–1,788081 rad; aynı indeksli referans farkları 0,184468–0,217792 rad. En büyük PD farkı B/Jacobian seçiminin sağ ayak bileği pitch ekleminde, kontrol150→151 / 3,02s. Seçili altının fark dizileri ham hedeflerden ayrıca doğrulandı. Başlangıç hedef–qpos farkı, tam koşu ölçümleri ve 0,4/2,0/3,6s çevresinde eş süreli ±0,16s pencereler ayrı kaydedildi. Bu post-hoc betimleme tork, jerk, görünen akıcılık veya nedensel yöntem üstünlüğü ölçümü değildir; yeni eşik/seçim kuralı uygulanmadı. Yeni fizik veya model çağrısı yapılmadı. Yeniden üretim aracı: [audit_pd_steps.py](audit_pd_steps.py); mevcut kanıt üzerine yazmayı reddeder.

## Kanıt ve kullanım

- [Protokol](../../artifacts/runs/agentic/multifeature_music_v1/protocol.json), [karşılaştırma](../../artifacts/runs/agentic/multifeature_music_v1/comparison.json), [semantik eşleştirme](../../artifacts/runs/agentic/multifeature_music_v1/semantic_evaluation.json).
- [Deney kodu](multifeature_driver.py), [hareket aracı](../../src/pulsemorph/agentic/multifeature_reference.py), [ortak oynatıcı](../../src/pulsemorph/agentic/common_player.py), [yürütücü](execute_multifeature.py), [ölçüm](evaluate_multifeature.py), [ses/görev hazırlığı](musical_tasks.py).
- `trials/` gerçek ONNX girdileri/çıktıları, ortak reference, PD hedefleri, qpos/qvel, her fizik adımı ve kaynak snapshot'larını içerir. `observations/`, `responses/`, `proposals/`, `trial_records/` karar soyunu korur.
- [60 hedefli testin kaydı](../../artifacts/runs/agentic/multifeature_music_v1/verification/targeted_tests.json): türev/kimlik/bounds, ortak save/load, aynı-zaman FK denetimi, fiziksel özelliklerin referanstan ayrılması, sekiz aday sınırı, eski/yabancı gözlem reddi ve gerçek PCM cue değişimi.
- [Son bağımsız denetim](../../artifacts/runs/agentic/multifeature_music_v1_audit_v1/audit.json) geçti:49 koşu yeniden ölçüldü;9.800 kararın gerçek ortak reference girdileri, aynı-zaman torso FK,48 önerinin kökeni, tüm save/load paketleri, ön taahhüt sırası ve son seçim doğrulandı. Altı akışın ilk aday qpos/qvel/PD izleri bütünüyle bit düzeyinde aynı. Denetim kaynakları ve özgün kanıtlar denetim boyunca değişmedi; frozen dosyalar korundu.
- [Birleştirilebilir veri için ek indeks](../../artifacts/runs/agentic/multifeature_music_v1_data_v1/index.json):49 kayıt,1 motion group,2 audio group,1.108 hash-bağlı dosya bağlantısı. Ortak corpus import'u değildir.
- Modelin allowed input'u audio feature JSON ve önceki tamamlanmış ayrı koşuların beden özetidir. Ses88200 örnek/4s offline erişim; önceki beden feedback'i kendi runtime ID'sinde4s'ye kadar gerçekleşmiş gözlemdir. Henüz çalıştırılmamış yeni take'in geleceği gözlenmiş sayılmaz. Runtime motoru50Hz gerçek mevcut beden durumunu görür.

[Görev A videosu](../../artifacts/media/agentic/multifeature_music_v2/swap_left_to_right/comparison.mp4) · [Görev B videosu](../../artifacts/media/agentic/multifeature_music_v2/swap_right_to_left/comparison.mp4).

Kaydedilmiş fizik durumlarından hazırlanan videolar yeni fizik replay'i değildir. Render yalnız `mj_forward` kullanır; her vaka100kare/25FPS/4s, kendi müziğiyle. FFprobe kare/saat/çözünürlük denetimleri geçti ve iki orta kare görsel incelendi. v2 kamera, ölçüm panelinin ayakları kapatmaması için uzaklaştırıldı; v1 korunur.

```powershell
$env:PYTHONPATH='src;.;experiments/agentic'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv-motor/Scripts/python -m pytest tests/test_agentic_multifeature_reference.py tests/test_agentic_common_player.py tests/test_agentic_multifeature_evaluation.py tests/test_agentic_multifeature_workflow.py -q
# Her çıktı yolu yeni olmalı; mevcut evidence üzerine yazılmaz.
.venv-motor/Scripts/python experiments/agentic/musical_tasks.py --output artifacts/runs/agentic/YENI_DENEY
# Gözlemlerden gerçek model yanıtları sağlanır; kaydedilmiş yanıt replay'i yeni model çağrısı sayılmaz.
.venv-motor/Scripts/python experiments/agentic/audit_multifeature.py --pilot artifacts/runs/agentic/multifeature_music_v1 --output artifacts/runs/agentic/YENI_AUDIT
.venv-motor/Scripts/python experiments/agentic/multifeature_preview.py --pilot artifacts/runs/agentic/multifeature_music_v1 --output artifacts/media/agentic/YENI_PREVIEW
```

Sonraki kanıt kapısı: aynı katkıyı farklı motion source_group'larda ve tutulmuş müzik/talimatlarda aramak; tepe seçimi sıçramalarını modelleyen güçlü optimizer ve model çağrı bütçesi eşlenmiş feedback ablation eklemek; algısal ifade ve beğeniyi bağımsız insan/VLM çalışmasında ayrı ölçmek.
