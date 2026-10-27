# High5_b yetenek tanığı aramalarının sonuçları

26 Ekim 2026. Aynı high5_b sayısal hedefi için global clock aracında 64 yeni aday çözüm bulamadı. Ardından ayrı dondurulmuş yerel clock aracıyla yapılan aramada 38. aday bütün hedefleri geçti; aynı parametreyle tek fresh-reset confirmation sonrasında yürütme özeti `confirmed_witness` oldu. Global ve yerel bağımsız denetimler geçti. Eski dört görevli LLM karşılaştırmasının 3/4 sonucu değiştirilmedi.

**Sonraki bağlam incelemesi:** Eski LLM karşılaştırmasının B gözlemlerinde kalibrasyona ait A-hedef hata vektörü taşındığı bulundu; eski3/4 sonucu bu tutarsız bilgi koşulunun betimleyici kaydıdır. Bu rapordaki sayısal kapasite aramaları doğru ham özellikleri ve güncel hedefi kullanır; yerel aday37/confirmation'ın ölçülmüş çözüm tanığı bu kusurla değişmez. [Kapsam incelemesi](../../design/agentic/CALIBRATION_CONTEXT_REVIEW.md). Önceki teslim manifesti bu sonraki açıklamadan önceki doküman snapshot'ıdır.

## Sabit hedef ve yürütme

Kullanıcı talimatı: “İlk bölümde sol kol yüzde 110, sağ kol yüzde 75 düzeyinde olsun; ikinci bölümde sol yüzde 75, sağ yüzde 110 olsun. Ölçüm penceresindeki baskın hareket vurgusunu müziğin en güçlü bas olayına hizala. Fiziksel geçerlilik öncelikli.” Bu aşamada bütün sayısal yöntemler aynı önceden belirlenmiş hedefi alır; yeni dil ayrıştırma veya müzik anlama başarısı ölçülmez. Vurgu hedefi, önceki görevin değişmemiş 4 saniyelik PCM dosyasında ölçülmüş `bass_peak_05` olayıdır. Sıfır fazlı offline ses analizi ve authored bölüm sınırları kullanılır; canlı nedensel müzik algısı değildir.

Özellik sırası sol-A, sağ-A, sol-B, sağ-B RMS oranı ve baskın vurgu zamanıdır. Hedef `[1.10,.75,.75,1.10,2.2948072562358277]`; oran toleransı ±.08, zaman toleransı ±.10 s. Bütünlük ve fizik kapısı: minimum root yüksekliği ≥.60 m, ortalama joint RMSE ≤.35 rad. Oranlar fresh gerçek fizik calibration'ına göre ölçülen named arm-joint displacement RMS'leridir; gain parametresi doğrudan gerçekleşen oran değildir.

Dört gain [.35,1.60], timing shift [−.30,.30] sınırındadır. Kaynak high5, G1 beden ve pretrained ONNX motor korunur; 4 saniyelik offline take, MotionReference üzerinden gerçek ONNX→PD→MuJoCo yürütmesidir. Global clock tüm 0..4 s içinde `t+d sin²(pi*t/4)`; yerel clock yalnız 1.4..2.8 s içinde `t+d sin²(pi*(t−1.4)/1.4)` ve dışarıda identity'dir. Bölüm gain envelope'ları değiştirilmedi.

## Ölçülen sonuçlar

| Koşu | Sol-A | Sağ-A | Sol-B | Sağ-B | Vurgu s | En kötü normalize hata | Beş hedef |
|---|---:|---:|---:|---:|---:|---:|---|
| Eski LLM feedback seçimi7 | 1.134504 | .746032 | .860887 | 1.108057 | 2.14 | 1.548073 | Geçmedi |
| Global yeni en iyi24 | 1.134162 | .696637 | .858350 | .992639 | 2.16 | 1.354374 | Geçmedi |
| Yerel aday37 | 1.114593 | .690326 | .814944 | 1.056179 | 2.22 | .811802 | Geçti |
| Yerel confirmation37 | 1.114593 | .690326 | .814944 | 1.056179 | 2.22 | .811802 | Geçti |

Yerel tanığın zaman hatası −.074807 s; sol-B hatası +.064944'tür. Tolerans içinde olmak tam hedefe ulaşmak demek değildir. Global en iyi koşunun sol-B hatası +.108350, zaman hatası −.134807 s ve sağ-B hatası −.107361 kalır. Seçim fizik kapısından sonra en büyük normalize hata, kare normalize hata toplamı ve kimlik sırasını kullanır; kalite diagnostikleri yeniden sıralamaz.

Parametre sırası `[left_gain_a,right_gain_a,left_gain_b,right_gain_b,timing_shift_s]`:

- Eski7: `[1.6,.85,.35,.95,−.1995]`.
- Global24: `[1.5999999999999996,.744532336230312,.35,.7604020477091179,−.20157728604876055]`.
- Yerel37 ve confirmation: `[1.1587112873069414,.5505062944838495,.35,.9099463244822236,−.27707405016186226]`.

Confirmation ilk geçen vektörün tek yeni reset tekrarıdır. Donmuş politika qpos/qvel/uygulanmış PD hedef dizilerinin tam byte eşitliğini ve bütün beş hedefin yeniden geçmesini ister; hata/fark olsaydı retry veya sonraki aday olmadan duracaktı. Bu koşullar bağımsız yerel v2 denetiminde de doğrulandı. Bu deterministik tekrar rastgele perturbation sağlamlığı ölçümü değildir.

Yerel arama içinde daha dar bir karşılaştırma da vardır: [aday22](../../artifacts/runs/agentic/high5_local_capability_v1/evaluations/high5_b__local_capability__22.json) ile [aday37](../../artifacts/runs/agentic/high5_local_capability_v1/evaluations/high5_b__local_capability__37.json) aynı dört gain değerini kullanır. Tek değişen aday parametresi shift'in −.235074'ten −.277074 s'ye inmesidir. Baskın gerçek vurgu 2.18'den2.22 s'ye, sol-B oranı .841030'dan .814944'e geçer; iki başarısız boyut toleransa girer. Aday37, donmuş aramanın `coordinate_probe` adımıdır; elle eklenmiş bir öneri değildir. Bu iki kayıt zaman düzenlemesinin gerçekleşen genliği de değiştirebildiğini gösterir; bütün parametre alanına ait türev veya global/yerel araç üstünlüğü sonucu değildir.

## Fizik ve kalite diagnostikleri

| Koşu | Min root m | Ortalama joint RMSE rad | PD adımı RMS rad | PD en büyük adım rad | 20 ms fark ivmesi RMS rad/s² |
|---|---:|---:|---:|---:|---:|
| Eski7 | .784499 | .134099 | .050201 | .792683 | 17.325038 |
| Global24 | .784598 | .134754 | .050187 | .792679 | 17.296043 |
| Yerel37 / confirmation | .786760 | .129769 | .054366 | .863085 | 18.493318 |

Bu seçilmiş koşular fizik kapısını geçti. Yerel PD adımları ve sonlu fark ivmesi daha yüksektir; bu betimleyici değişiklik tek başına daha iyi/kötü dans kararı vermez. 20 ms örnekleme kontrol arası ivmeyi atlar. Tork, ayak teması/slip, recovery, algılanan ifade, insan tercihi veya donanım güvenliği bu kapılarla doğrulanmaz. Root displacement farkı da bağımsız rota takip ölçütü değildir.

## Bütçe ve karşılaştırılabilirlik

| Aşama | Yeni calibration | Yeni aday | Confirmation | Yeni gerçek koşu | Kontrol adımı | Fizik adımı |
|---|---:|---:|---:|---:|---:|---:|
| Global | 1 | 64 | 0 | 65 | 13.000 | 260.000 |
| Yerel | 1 | 38 | 1 | 40 | 8.000 | 160.000 |
| Toplam | 2 | 102 | 1 | 105 | 21.000 | 420.000 |

Her koşu 200 kontrol/4.000 fizik adımıdır. Özetlerde toplam 21.000 online ONNX çağrısı ve ayrıca105 warmup çağrısı vardır. Yeni LLM proposal çağrısı **0**; bu sayısal arama, yeni planner karşılaştırması değildir. Yerel tanık confirmation'ı geçince kalan 26 aday slotu tüketilmedi.

Global arama önceki high5_b karşılaştırmasından 24 ölçüm satırını (20 benzersiz vektör) kullandı; bu eski üretim maliyeti yukarıdaki105 yeni koşuya dahil değildir. Eski prior ancak yeni calibration'ın tam qpos/qvel/PD byte eşitliği ve pin doğrulaması sonrası kullanıldı. Yerel arama **0 eski feature label** içeri aldı; kendi1 fresh calibration'ı tek ölçülmüş başlangıç prior'ıdır ve40 koşuya dahildir. Ayrıca yerel seed/probe takvimi globalden farklıdır. Dolayısıyla sonuçlar eşlenmiş bir araç ablation'ı veya clock desteğinin tek başına nedensel faydasının ölçümü değildir.

## Denetim ve kayıt sınırları

[Global audit](../../artifacts/runs/agentic/high5_capability_v1_audit_v1/audit.json) ve [yerel v2 audit](../../artifacts/runs/agentic/high5_local_capability_v1_audit_v2/audit.json) `audit_passed=true` bildirir. 105 yeni koşunun tümü bütünlük ve yerel fizik kapısını geçti; beş talimat hedefini geçen iki kayıt aynı yerel aday ve onun tekrarıdır. 105 ortak MotionReference save/load kontrolü ile 21.000 gerçek kararın referans girdileri doğrulandı; en büyük mutlak input farkı0. Aynı-zaman torso FK en büyük açı hatası globalde4,23e−8 rad, yerelde2,99e−8 rad. Fizik yeniden koşturulmadı; kaydedilmiş durumlarda FK ve metrikler yeniden hesaplandı. Her iki denetimde özgün deney verileri değişmedi. Yerel denetim24 kaynak dosyasını önce/sonra ve snapshot ile sabitledi; global denetim kaynakları ise yürütmeden sonra ayrı companion inventory'de yakalandı, aynı düzeyde önce/sonra kaynak envanteri iddiası yoktur.

[İlk yerel audit](../../artifacts/runs/agentic/high5_local_capability_v1_audit_v1/audit.json), denetleyicinin eski `generalization-surrogate_search` üretici adını beklemesi nedeniyle durdu. Donmuş yerel runtime'ın doğru adı `local-capability-surrogate_search` idi. Yalnız denetleyici düzeltildi; manifest ile ortak MotionReference üreticisinin birebir aynı yerel kimliği taşıması da kontrol ediliyor. Hatalı global kimliği reddeden gerçek runtime/kayıt entegrasyon testi eklendi. Başarısız audit, kaynak snapshot'ı ve bütün özgün fizik verileri korundu; başarı yeni v2 dizininde kaydedildi.

Global arama/workflow test kaydı17 geçiştir; ek6 audit testi ayrı bileşen doğrulamasıdır. Yerel tool/workflow test kaydı31 geçiştir. Yerel audit yardımcılarının ilk5 testi geçti; yukarıdaki entegrasyon eksiği eklenip beklenti düzeltildikten sonra [6 testin tamamı geçti](../../artifacts/runs/agentic/local_capability_audit_fix_v1/test_run.json). İlk5 ile son6 aynı testlerin iki sürümüdür;11 bağımsız test diye toplanmaz. Bu sayılar fizik koşusu veya tek bir birleşik repo testi çalışması değildir.

Global legacy proposal template'inde `trial_index=0` kalmıştır. Orijinal kanıt değiştirilmeden eklenen [canonical annotation](../../artifacts/runs/agentic/high5_capability_v1/annotations/canonical_trial_indices_v1.json), kimlik ve `search_algorithm.new_candidate_index` üzerinden gerçek0..63 sırasını açıklar. Yerel proposal'da gerçek aday indeksi37 kaydedilmiştir. Bu annotation ölçüm düzeltmesi veya eğitim kabulü değildir. Audit'lerde `training_ready=false`; bu yeni kayıtlar ortak eğitim corpus'una alınmadı ve train/dev/test ataması yapılmadı.

Kaynak/protokol/model/profile ve evaluator hashleri donmuş kayıtların parçasıdır; listelenmiş pin ile bağımsız doğrulanmış pin ayrımı audit kapsamındadır. Bir dosyanın mevcut olması, yürütülmesi, bağımsız audit geçmesi ve ürün kullanımına uygunluğu ayrı kanıt düzeyleridir.

## Video ve yorum sınırı

[Global karşılaştırma videosu](../../artifacts/media/agentic/high5_capability_v1/comparison.mp4) ve [yerel çözüm/tekrar videosu](../../artifacts/media/agentic/high5_local_capability_v1/comparison.mp4) hazırdır. Her ikisi 960×720, 100 kare, 25 FPS, 4 saniye; 4 saniyelik görev sesiyle FFprobe doğrulaması geçti. İki orta kare görsel olarak incelendi: beden/ayaklar görünür, etiketler okunabilir. Görsel kontrol tek örnek kareyle sınırlıdır; bütün karelerde algısal kalite doğrulaması değildir. Videolar gerçek fizik sırasında kaydedilmiş qpos değerlerinin `mj_forward` render'ıdır; yeni fizik koşusu veya motor replay'i değildir. Global videoda bulunmayan confirmation açıkça boş bilgi paneliyle gösterilir; yerel videoda bağımsız resetten alınmış gerçek confirmation vardır. Eski LLM sonucu farklı bütçeli tarihsel karşılaştırmadır.

Global aramanın sonlu başarısızlığı global araçta çözümün olanaksızlığını göstermez. Yerel tanık bu kaynak/hedef/beden/motor/araç altında en az bir tekrar edilmiş çözüm bulunduğunu gösterir; genel yetenek, öğrenme, estetik, canlı recovery, beden aktarımı veya ürün kalitesi çıkarımı vermez. Eski LLM'nin3/4 sonucu, aynı sekizli çağrı bütçesinin tarihsel sonucu olarak korunur.

Ana kanıtlar: [global summary](../../artifacts/runs/agentic/high5_capability_v1/summary.json), [yerel summary](../../artifacts/runs/agentic/high5_local_capability_v1/summary.json), iki donmuş protocol, seçilmiş evaluations ve confirmation manifesti. Referans/gerçekleşen RMS ayrımı için [ayrı diagnostik](../../design/agentic/HIGH5_CAPABILITY_DIAGNOSIS.md) bulunur.

Sonraki ayırt edici deney için [iki araçla LLM seçimi ve geri bildirim taslağı](../../design/agentic/LLM_TOOL_SELECTION_NEXT.md) hazırlandı. Bu taslak henüz dondurulmadı veya çalıştırılmadı; mevcut high5_b geliştirme verisi olarak kalır. Aynı iki aracın tüm yöntemlere sunulması, görülmemiş görev seçiminin önceden sabitlenmesi ve eşit fizik/LLM çağrı bütçesi önerilir. Bugünkü sayısal çözüm LLM'nin araç seçimi katkısının kanıtı değildir.
