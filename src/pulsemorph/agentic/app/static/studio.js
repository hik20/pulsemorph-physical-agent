'use strict';
fetch('/api/phrase/status').then(response => {
  if (!response.ok) throw new Error('status unavailable');
  return response.json();
}).then(status => {
  const note = document.getElementById('studio-status');
  if (status.legacy_disk?.available === false) {
    note.textContent = '4 saniyelik çalışma deposunda en az 256 MiB boş alan gerekiyor. Bu modda yeni deneme başlatılmadan alan kontrol edilir. 8 saniyelik akış ayrı depoyu kullanır.';
  } else {
    note.textContent = 'İki akış arasında aynı anda bir çalışma yürütülür. Bir akışta deneme sürerken diğerinde yeni deneme başlatılamaz.';
  }
}).catch(() => {
  document.getElementById('studio-status').textContent = 'Yerel durum okunamadı. Stüdyoyu açarak kurulum ve kaynak durumunu kontrol edebilirsin.';
});
