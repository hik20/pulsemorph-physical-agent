# İki kaynakta agentic çok özellikli deney

**26 Ekim 2026 ek kapsam düzeltmesi:** Sonraki incelemede B görevlerinin LLM gözlemlerinde, ortak kalibrasyonun doğru ham ölçümlerinin yanında A hedeflerine göre hesaplanmış `feature_errors` bulundu. 32 observation bu tutarsız alanı taşıyor. 96 adayın kendi hedeflerine göre hesaplanmış skorlarında aritmetik uyuşmazlık yok; aşağıdaki 3/4–1/4–0/4 sayıları kayıtlı gerçekleşmeleri korur. Ancak bunlar tutarsız başlangıç bağlamı altında alınmıştır; temiz koşulda LLM/geri bildirim katkısı olarak yorumlanmamalıdır. Hatanın kararları ne ölçüde etkilediği bilinmiyor. [Ayrıntılı kapsam incelemesi](../../design/agentic/CALIBRATION_CONTEXT_REVIEW.md). Özgün protokol, yanıtlar, fizik ve denetim kayıtları değiştirilmedi.

Bu rapor tamamlanmış deney ve geçen bağımsız kayıt auditinden üretilmiştir; yeni fizik veya model çağrısı içermez.

**Sonuç:** Dört kol hedefini ve müzik vurgusunu birlikte sağlayan görev sayısı LLM + fizik geri bildiriminde **3/4**, eşit öneri turu sayılı geri bildirimsiz LLM'de **1/4**, RBF aramasında **0/4**. High5_b üç yöntemde de çözülemedi. Bu, iki kaynak dosyasındaki dört ilişkili görevin ölçümüdür; istatistiksel veya genel estetik üstünlük sonucu değildir.

Wave_b'de iki LLM koşulu da ilk kez beşinci fizik adayında geçti; geri bildirimli modelin sekizinci adayı dört kol oranını hedeflere 0,0002'den küçük mutlak hatayla, vurguyu 2,9ms hatayla getirdi. Wave_a ancak sekizinci adayda geçti ve zaman hatası 93,2ms ile 100ms sınırına yakındır. High5_b'de seçili LLM'nin sol-B oranı 0,861 yerine 0,750 hedefini ve 154,8ms zaman hatasıyla vurgu toleransını sağlayamadı. Sonlu aramanın bu başarısızlığı hedefin olanaksızlığını kanıtlamaz.

Gerçek bütçe: **98 fizik koşusu, 19600 kontrol kararı, 392000 MuJoCo adımı**; 96 aday + 2 kaynak kalibrasyonu.

Her koşuda bir warmup ONNX çağrısı doğrulandı: toplam 98 warmup, 19.600 aktif kontrol çağrısından ayrıdır. Gerçekleşen kontrol örnekleme aralığı 20 ms (50 Hz) olarak doğrulandı.

Dört görev, iki motion dosyası/kaynak grubu üzerinde ilişkilidir; aynı G1 beden, pretrained motor ve generated kaynak ailesini kullanır. Cue yerleşimi fizik öncesi kaynak bilgisiyle hazırlanmıştır; bu saf kör kaynak değerlendirmesi veya bağımsız insan genellemesi değildir.

## Görevler ve hedefler

Oran sırası L_A/R_A; L_B/R_B (sol/sağ, bölüm A; sol/sağ, bölüm B). Zaman hedefleri gerçek waveform analizinden gelen kanonik değerlerdir.

| Görev | Kaynak | Hedef oranlar | Hedef vurgu s |
|---|---|---|---:|
| wave_a | wave | 0.800/1.200; 1.200/0.800 | 2.993220 |
| wave_b | wave | 1.100/0.750; 0.750/1.100 | 3.122925 |
| high5_a | high5 | 0.800/1.200; 1.200/0.800 | 2.175079 |
| high5_b | high5 | 1.100/0.750; 0.750/1.100 | 2.294807 |

Mutlak toleranslar: L_A=0.080, R_A=0.080, L_B=0.080, R_B=0.080; vurgu zamanı ±100.0 ms.

## Seçilen adaylar

Oranlar gerçekleşen fiziksel eklem hareketinin aynı kaynağın değişmemiş fizik kalibrasyonuna oranıdır. Sıra sol-A, sağ-A, sol-B, sağ-B; zaman hatası baskın hareket vurgusu ile ölçülen müzik cue’sunun mutlak farkıdır.

| Görev | Yöntem | Aday indeksi | L_A/R_A; L_B/R_B | Zaman hatası ms | Tüm hedefler | En kötü normalize hata |
|---|---|---:|---|---:|---|---:|
| wave_a | RBF araması | 7 | 0.900/1.150; 1.273/0.910 | 93.2 | hayır | 1.370 |
| wave_a | LLM + geri bildirim | 7 | 0.852/1.200; 1.201/0.847 | 93.2 | evet | 0.932 |
| wave_a | LLM geri bildirim kapalı | 6 | 0.935/1.153; 1.146/0.883 | 93.2 | hayır | 1.686 |
| wave_b | RBF araması | 2 | 1.092/0.725; 0.794/1.090 | 222.9 | hayır | 2.229 |
| wave_b | LLM + geri bildirim | 7 | 1.100/0.750; 0.750/1.100 | 2.9 | evet | 0.029 |
| wave_b | LLM geri bildirim kapalı | 5 | 1.130/0.758; 0.757/1.112 | 22.9 | evet | 0.377 |
| high5_a | RBF araması | 6 | 0.651/1.166; 1.062/0.836 | 124.9 | hayır | 1.865 |
| high5_a | LLM + geri bildirim | 7 | 0.801/1.171; 1.193/0.839 | 55.1 | evet | 0.551 |
| high5_a | LLM geri bildirim kapalı | 0 | 1.000/1.000; 1.000/1.000 | 215.1 | hayır | 2.500 |
| high5_b | RBF araması | 7 | 0.949/0.849; 0.908/1.063 | 294.8 | hayır | 2.948 |
| high5_b | LLM + geri bildirim | 7 | 1.135/0.746; 0.861/1.108 | 154.8 | hayır | 1.548 |
| high5_b | LLM geri bildirim kapalı | 0 | 1.000/1.000; 1.000/1.000 | 334.8 | hayır | 3.348 |

Aday indeksi 0–7 aralığındadır; 0 değişmemiş anchor’dır. Tam aday kimlikleri bağlantılı comparison.json içinde korunur. Seçim fizik kapısından geçen adaylar arasında en büyük mutlak tolerans hatası, sonra normalize hata kareleri toplamı ve aday kimliği sırasındadır. Bu selector genel dans kalitesi puanı değildir; kalite diagnostikleri seçimi yeniden sıralamaz.

## Yöntem ve bütçe özeti

| Yöntem | Seçilen adayla başarılı görev / 4 | Başarılı aday / 32 | Fizik kapısı / 32 | İlk başarılı deneme (1–8, anchor dahil) |
|---|---:|---:|---:|---|
| RBF araması | 0/4 | 0/32 | 32/32 | wave_a: yok; wave_b: yok; high5_a: yok; high5_b: yok |
| LLM + geri bildirim | 3/4 | 10/32 | 32/32 | wave_a: 8; wave_b: 5; high5_a: 4; high5_b: yok |
| LLM geri bildirim kapalı | 1/4 | 2/32 | 32/32 | wave_a: yok; wave_b: 5; high5_a: yok; high5_b: yok |

Tüm koşularda fizik kapısı: 98/98; kayıt bütünlüğü: 98/98. İki kalibrasyon intent yarışmasına dahil değildir.

Her LLM yönteminde her görev için sekiz ayrı model yanıtı/turu vardır: feedback 32, feedback-off 32, toplam 64. Çağrı sayıları eşleştirilmiştir; token, fatura, bağlam uzunluğu veya izole model gecikmesi eşitliği gösterilmemiştir. OFF için 32 önerinin tamamı aday fiziğinden önce taahhüt edilmiştir; feedback yalnız kendi önceki sonuçlarını görür. Gözlem kısıtları prompt ve ayrı fork/agent akışlarına dayanır; işletim sistemi düzeyinde sandbox izolasyonu iddiası değildir.

## Ayrı kalite diagnostikleri

Aşağıda her hücre seçilen aday / aynı kaynak kalibrasyonu değeridir. Önceden tanımlanmış kalite kabul eşiği yoktur; yüksek veya düşük değer tek başına iyi dans anlamına gelmez.

| Görev / yöntem | PD hedef adımı RMS / max rad | Gerçek hızdan FD ivme RMS / max rad/s² | Root displacement farkı RMS / son m |
|---|---|---|---|
| wave_a / RBF araması | 0.040 / 0.032; 0.793 / 0.751 | 14.790 / 10.869; 384.115 / 116.457 | 0.070 / 0.045; 0.080 / 0.041 |
| wave_a / LLM + geri bildirim | 0.039 / 0.032; 0.751 / 0.751 | 14.307 / 10.869; 325.915 / 116.457 | 0.066 / 0.045; 0.077 / 0.041 |
| wave_a / LLM geri bildirim kapalı | 0.040 / 0.032; 0.825 / 0.751 | 15.182 / 10.869; 406.231 / 116.457 | 0.073 / 0.045; 0.084 / 0.041 |
| wave_b / RBF araması | 0.031 / 0.032; 0.751 / 0.751 | 10.566 / 10.869; 116.513 / 116.457 | 0.040 / 0.045; 0.047 / 0.041 |
| wave_b / LLM + geri bildirim | 0.030 / 0.032; 0.751 / 0.751 | 10.059 / 10.869; 116.457 / 116.457 | 0.042 / 0.045; 0.045 / 0.041 |
| wave_b / LLM geri bildirim kapalı | 0.030 / 0.032; 0.751 / 0.751 | 10.142 / 10.869; 116.457 / 116.457 | 0.042 / 0.045; 0.045 / 0.041 |
| high5_a / RBF araması | 0.045 / 0.049; 0.793 / 0.793 | 14.101 / 17.420; 200.168 / 274.137 | 0.080 / 0.086; 0.070 / 0.084 |
| high5_a / LLM + geri bildirim | 0.049 / 0.049; 0.793 / 0.793 | 16.684 / 17.420; 274.058 / 274.137 | 0.098 / 0.086; 0.114 / 0.084 |
| high5_a / LLM geri bildirim kapalı | 0.049 / 0.049; 0.793 / 0.793 | 17.420 / 17.420; 274.137 / 274.137 | 0.086 / 0.086; 0.084 / 0.084 |
| high5_b / RBF araması | 0.047 / 0.049; 0.793 / 0.793 | 16.922 / 17.420; 277.827 / 274.137 | 0.080 / 0.086; 0.079 / 0.084 |
| high5_b / LLM + geri bildirim | 0.050 / 0.049; 0.793 / 0.793 | 17.325 / 17.420; 238.587 / 274.137 | 0.086 / 0.086; 0.094 / 0.084 |
| high5_b / LLM geri bildirim kapalı | 0.049 / 0.049; 0.793 / 0.793 | 17.420 / 17.420; 274.137 / 274.137 | 0.086 / 0.086; 0.084 / 0.084 |

PD adımı uygulanmış hedef farkıdır. İvme 20 ms kontrol sınırlarında gerçekleşen qvel farkından hesaplanır; adım içi ivmeyi ölçmez. Root farkı unwarped kaynak ile source-world displacement karşılaştırmasıdır; root rota kontrolü veya kayma mesafesi değildir. Temas/slip, insan beğenisi ve algısal ifade kalite kabulü ölçülmemiştir.

[Ayrı sonuç incelemesinde](../../design/agentic/GENERALIZATION_REVIEW.md) seçili dört LLM adayının en büyük PD adımı 0,02s başlangıç geçişinde, waist_pitch_joint üzerinde bulundu. Tam koşu maksimumunun benzer olması edit bölümünün aynı akıcılıkta olduğunu göstermez. Wave_a hedefleri geçerken örneklenmiş gerçek ivme RMS'i 10,869→14,307rad/s² yükseldi; hedef başarısı bütün yan etkilerin iyileşmesi değildir.

## Kapsam ve köken

Bütün yöntemlere aynı özgün kanonik hedef verildi; ayrı semantik aşama yoktur. Model gözlemi waveform analizinden sayısal müzik özellikleri alır; ham PCM’i duyduğu veya genel müzik algısını çözdüğü iddia edilmez. Hedeflerin fiziksel erişilebilirliği bilinmemektedir; sekiz adayda başarısızlık görevin olanaksızlığını kanıtlamaz.

Bu deney canlı recovery, eğitimle ağırlık geliştirme, güvenli donanım kontrolü, üretim kabulü veya genel dans kalitesi sonucu vermez. Eski v1’in 2/2 geliştirme görevi sonucu yalnız tarihsel bağlamdır; yeni görevler ve kaynaklarla istatistiksel karşılaştırma değildir.

Protokol: [protocol.json](<../../artifacts/runs/agentic/generalization_music_v1/protocol.json>); SHA256 `80c2fac7bbc1a886e774e4d32aa569c069a3f4750e91a4a33628f8594df667ec`. Karşılaştırma: [comparison.json](<../../artifacts/runs/agentic/generalization_music_v1/comparison.json>); SHA256 `7008863795335265502ad6c8ddf8b9d0a4136c81550151cc859f1c1fab6a3d11`.

Bağımsız audit: [audit.json](<../../artifacts/runs/agentic/generalization_music_v1_audit_v1/audit.json>); SHA256 `4db2798f104a0b4fcc3411d80f0be12377c02a87bc7e6bd61259b0423d6192ca`; audit_passed=true. Kaynak kod pinleri protokolün execution_code_sha256 alanında; audit kaynak pinleri audit dosyasında kayıtlıdır ve rapor üretiminde yeniden kontrol edilir.

Denetim dört PCM gözlemini yeniden ölçtü; 98 ortak referans paketini yeniden yükleyip örnekledi. 19.600 gerçek kontrol kararında ortak reference girdilerinin en büyük farkı 0; aynı-zaman torso FK açısal farkı en çok 4,69e-8rad. Her kaynaktaki altı akışın ilk aday qpos/qvel/PD dizileri kalibrasyonla bit düzeyinde aynı. Bütün özgün kanıtlar ve korunan dosyalar değişmedi. Bu kayıt denetimi bağımsız bir dinamik replay veya insan değerlendirmesi değildir.

Yeni runtime/search/workflow için **35 hedefli test**, bağımsız denetim yardımcıları için **11 test** geçti. Bütün depo testleri yeniden çalıştırılmış sayılmaz. LLM önerileri Codex alt agentlarıyla gerçek model turlarıdır; bağımsız sağlayıcı API servisi veya dağıtılabilir ürün tamamlanmış değildir. Yeni ağırlık eğitimi ve ortak corpus alımı yapılmadı; kayıtlar `unassigned` / `unassessed` / `training_admission=false` kalır.

| Kaynak grubu | Motion yolu | SHA256 |
|---|---|---|
| high5 | [output_high5.motion](<../../research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_high5.motion>) | `ef666262b8cccfd5279ce07b84eba9684e39ba2ca1a1cd1e2b55715a0394cf44` |
| wave | [output_wave.motion](<../../research/vendor/ProtoMotions/data/g1-kimodo-generated/proto/output_wave.motion>) | `cb23fb229ca49116462624a8164d1755eff1c2e57dce75dfd896a1c24113bc93` |

## Videolar

Videolar gerçekleşen kayıtları incelemek içindir; ayrı insan kalite değerlendirmesi değildir.

- wave_a: [comparison.mp4](<../../artifacts/media/agentic/generalization_music_v1/wave_a/comparison.mp4>)
- wave_b: [comparison.mp4](<../../artifacts/media/agentic/generalization_music_v1/wave_b/comparison.mp4>)
- high5_a: [comparison.mp4](<../../artifacts/media/agentic/generalization_music_v1/high5_a/comparison.mp4>)
- high5_b: [comparison.mp4](<../../artifacts/media/agentic/generalization_music_v1/high5_b/comparison.mp4>)
