"""Local PCM WAV import and immutable four-second clips, composed with originals."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import unicodedata
import uuid
import wave

import numpy as np
from scipy.signal import resample_poly

from .music import AppMusic, RATE, _analyze

MAX_BYTES = 32 * 1024 * 1024
_ID = re.compile(r"upload_[0-9a-f]{24}")
_HEX = re.compile(r"[0-9a-f]{64}")
_LOCK = threading.RLock()
_TRANSFORM = "pcm16_downmix_resample_poly_48k_4s_v1"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _identity(source_sha256, start_frame):
    return "upload_" + _digest(_canonical({"source_sha256": source_sha256, "start_frame": start_frame, "transform": _TRANSFORM}))[:24]


def _label(value):
    if not isinstance(value, str):
        raise ValueError("Müzik etiketi metin olmalı")
    value = value.strip() or "Uploaded music"
    if len(value) > 80 or any(unicodedata.category(c).startswith("C") or c in "/\\" for c in value):
        raise ValueError("Müzik etiketi en fazla 80 düz metin karakteri olmalı; yol veya kontrol karakteri kullanmayın")
    return value


def _decode(data):
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_BYTES:
        raise ValueError("En fazla 32 MiB boyutunda PCM16 WAV yükleyin")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE" or struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("WAV başlığı veya dosya uzunluğu geçersiz; kesilmiş dosya kabul edilmez")
    chunks, offset = {}, 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("WAV bölüm başlığı eksik")
        name, size = data[offset:offset + 4], struct.unpack_from("<I", data, offset + 4)[0]
        begin, end = offset + 8, offset + 8 + size
        if end > len(data) or end + (size % 2) > len(data):
            raise ValueError("WAV ses verisi kesilmiş")
        if name in (b"fmt ", b"data"):
            if name in chunks:
                raise ValueError("Tek bir WAV biçim ve ses bölümü gerekli")
            chunks[name] = (begin, size)
        offset = end + size % 2
    if b"fmt " not in chunks or b"data" not in chunks or chunks[b"fmt "][1] < 16:
        raise ValueError("WAV biçim veya ses bölümü eksik")
    kind, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", data, chunks[b"fmt "][0])
    if kind != 1 or bits != 16 or channels not in (1, 2) or not 8000 <= rate <= 96000:
        raise ValueError("Yalnız 8–96 kHz, mono/stereo, sıkıştırılmamış PCM16 WAV destekleniyor")
    if align != channels * 2 or byte_rate != rate * align or chunks[b"data"][1] % align:
        raise ValueError("WAV örnek hizası veya örnek sayısı geçersiz")
    frames = chunks[b"data"][1] // align
    if not 4 * rate <= frames <= 120 * rate:
        raise ValueError("WAV süresi en az 4 ve en fazla 120 saniye olmalı")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getnframes() != frames or wav.getcomptype() != "NONE":
                raise ValueError("WAV kare sayısı başlıkla uyuşmuyor")
            raw = wav.readframes(frames)
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError("WAV dosyası okunamadı") from exc
    if len(raw) != frames * align:
        raise ValueError("WAV gerçek ses örnekleri eksik")
    return np.frombuffer(raw, dtype="<i2").reshape(frames, channels), rate, channels, frames


class StudioMusic:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[4]).resolve()
        self.originals = AppMusic(self.root)
        self.directory = self.root / "artifacts/media/agentic/app_user_music_v1"
        self._warnings = []

    def _paths(self, identity):
        if not _ID.fullmatch(identity):
            raise KeyError("Yüklenen müzik bulunamadı")
        if self.directory.resolve() != self.directory:
            raise ValueError("Müzik deposu beklenen yerel dizinin dışında")
        wav, manifest = self.directory / (identity + ".wav"), self.directory / (identity + ".json")
        if wav.is_symlink() or manifest.is_symlink():
            raise ValueError("Müzik dosyalarında sembolik bağlantı kabul edilmez")
        return wav, manifest

    def _load(self, identity):
        wav, manifest = self._paths(identity)
        if not wav.exists() and not manifest.exists():
            raise KeyError("Yüklenen müzik bulunamadı")
        try:
            if not wav.is_file() or not manifest.is_file() or manifest.stat().st_size > 100_000 or wav.stat().st_size > 400_000:
                raise ValueError("Yüklenen müzik kaydı eksik veya geçersiz")
            record = json.loads(manifest.read_text(encoding="utf-8"))
            checksum = record.pop("descriptor_sha256")
            if checksum != _digest(_canonical(record)) or record["id"] != identity or record["schema"] != "pulsemorph.uploaded_music.v1":
                raise ValueError("Yüklenen müzik kaydının bütünlüğü bozulmuş")
            source = record["provenance"]
            if not _HEX.fullmatch(source["source_sha256"]) or type(source["start_frame"]) is not int or source["start_frame"] < 0:
                raise ValueError("Yüklenen müzik kaynak kaydı geçersiz")
            if _identity(source["source_sha256"], source["start_frame"]) != identity or source["transform"] != _TRANSFORM:
                raise ValueError("Müzik kimliği kaynak kaydıyla uyuşmuyor")
            rate, channels, frames = source["original_sample_rate"], source["original_channels"], source["original_frame_count"]
            if type(rate) is not int or not 8000 <= rate <= 96000 or type(channels) is not int or channels not in (1, 2) or type(frames) is not int or not 4 * rate <= frames <= 120 * rate:
                raise ValueError("Yüklenen müzik kaynak ölçüleri geçersiz")
            if source["read_frame_count"] != frames or source["selected_frame_count"] != 4 * rate or source["start_frame"] + 4 * rate > frames or source["source_bytes_retained"] is not False:
                raise ValueError("Yüklenen müzik kesit kaydı geçersiz")
            if source["start_s"] != source["start_frame"] / rate or source["end_s"] != (source["start_frame"] + 4 * rate) / rate:
                raise ValueError("Yüklenen müzik kesit zamanı geçersiz")
            if record["label"] != _label(record["label"]) or record.get("path") is not None or record["url"] != f"/media/music/{identity}.wav":
                raise ValueError("Yüklenen müzik görüntüleme kaydı geçersiz")
            if record["bpm"] is not None or record["beat_times_s"] != [] or record["sha256"] != _digest(wav.read_bytes()):
                raise ValueError("Yüklenen müzik WAV özeti uyuşmuyor")
            if source["derived_wav_sha256"] != record["sha256"] or source["derived_frame_count"] != 4 * RATE or type(source["source_bytes"]) is not int or not 44 <= source["source_bytes"] <= MAX_BYTES:
                raise ValueError("Yüklenen müzik kaynak veya çıktı özeti geçersiz")
            measured = _analyze(wav)
            if measured["duration_s"] != 4 or any(record.get(key) != value for key, value in measured.items()):
                raise ValueError("Yüklenen müzik ölçümü WAV ile uyuşmuyor")
            if measured["rms"] <= .000001:
                raise ValueError("Sessiz müzik kesiti kullanılamaz")
            return dict(record, path=str(wav))
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError, OSError, wave.Error, OverflowError) as exc:
            raise ValueError("Yüklenen müzik kaydı doğrulanamadı; dosyalar değiştirilmedi") from exc

    def catalog(self):
        with _LOCK:
            records = self.originals.catalog()
            self._warnings = []
            if self.directory.exists():
                if self.directory.resolve() != self.directory:
                    self._warnings.append({"id": "upload_store", "message": "Yüklenen müzik deposunun yolu doğrulanamadı; kayıtlar açılmadı"})
                    return records
                identities = {path.stem for path in self.directory.glob("*") if path.suffix in (".wav", ".json") and _ID.fullmatch(path.stem)}
                for identity in sorted(identities):
                    try:
                        record = self._load(identity)
                        records.append({key: value for key, value in record.items() if key != "path"})
                    except (ValueError, KeyError, OSError):
                        self._warnings.append({"id": identity, "message": "Yüklenen ses kaydı eksik veya bozuk; dosyalar korundu ve bu kayıt kullanıma alınmadı"})
                staged = set()
                for path in self.directory.glob(".stage_*"):
                    match = re.search(r"upload_[0-9a-f]{24}", path.name)
                    staged.add(match[0] if match else "upload_staging")
                for identity in sorted(staged):
                    self._warnings.append({"id": identity, "message": "Tamamlanmamış yüklemenin hazırlık dosyaları inceleme için korundu; ses kataloğuna alınmadı"})
            return records

    def status(self):
        with _LOCK:
            records = self.catalog()
            return {"warnings": [dict(item) for item in self._warnings],
                    "uploaded_count": sum(item["id"].startswith("upload_") for item in records),
                    "full_source_bytes_retained": False}

    def get(self, identity):
        if not isinstance(identity, str):
            raise KeyError("Müzik bulunamadı")
        with _LOCK:
            return self._load(identity) if identity.startswith("upload_") else self.originals.get(identity)

    def import_wav(self, data: bytes, label: str, start_s: float):
        label = _label(label)
        if isinstance(start_s, bool) or not isinstance(start_s, (float, int)) or not math.isfinite(start_s) or start_s < 0:
            raise ValueError("Kesit başlangıcı sonlu, sıfır veya pozitif bir saniye olmalı")
        samples, rate, channels, frames = _decode(data)
        if start_s + 4 > frames / rate:
            raise ValueError("Seçilen başlangıçtan sonra dört tam saniye bulunmalı; sessizlik eklenmez")
        begin = round(start_s * rate)
        if begin + 4 * rate > frames:
            raise ValueError("Seçilen kesitin kare sayısı yetersiz")
        mono = samples[begin:begin + 4 * rate].astype(np.float64).mean(axis=1)
        common = math.gcd(rate, RATE)
        resampled = resample_poly(mono, RATE // common, rate // common)
        if len(resampled) != 4 * RATE or not np.isfinite(resampled).all():
            raise ValueError("Müzik yeniden örneklemesi doğrulanamadı")
        pcm = np.clip(np.rint(resampled), -32768, 32767).astype("<i2")
        if not np.any(np.abs(pcm.astype(np.int32)) > 1):
            raise ValueError("Seçilen müzik kesiti sessiz; ses içeren başka bir kesit seçin")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setparams((1, 2, RATE, 4 * RATE, "NONE", "not compressed"))
            output.writeframes(pcm.tobytes())
        clip, source_hash = buffer.getvalue(), _digest(data)
        identity = _identity(source_hash, begin)
        with _LOCK:
            wav, manifest = self._paths(identity)
            if manifest.exists():
                existing = self._load(identity)
                if existing["sha256"] != _digest(clip) or existing["provenance"]["source_sha256"] != source_hash or existing["provenance"]["start_frame"] != begin:
                    raise ValueError("Aynı kimliğin türetilmiş ses özeti değişti; önceki dosyalar korundu")
                return existing
            # A crash can publish the canonical WAV before the manifest commit.
            # Reimport may finish only when supplied bytes independently reproduce it.
            if wav.exists() and (not wav.is_file() or wav.read_bytes() != clip):
                raise ValueError("Yarım yüklemenin ses dosyası beklenen özetle uyuşmuyor; önceki dosya korundu")
            self.directory.mkdir(parents=True, exist_ok=True)
            stage = self.directory / (".stage_" + identity + "_" + uuid.uuid4().hex)
            stage_wav, stage_manifest = stage.with_suffix(".wav"), stage.with_suffix(".json")
            with stage_wav.open("xb") as target:
                target.write(clip)
            record = {"schema": "pulsemorph.uploaded_music.v1", "id": identity, "label": label,
                      "description": "User-provided four-second WAV segment; measured PCM accent, BPM unknown.",
                      **_analyze(stage_wav), "sha256": _digest(clip), "url": f"/media/music/{identity}.wav",
                      "bpm": None, "bpm_source": "unknown; tempo inference not performed", "beat_times_s": [],
                      "beat_times_source": "unknown; measured transients are onset_times_s",
                      "rights": "User-provided audio; ownership and redistribution license unknown. Local processing only.",
                      "provenance": {"source_sha256": source_hash, "source_bytes": len(data), "source_bytes_retained": False,
                          "original_frame_count": frames, "original_sample_rate": rate, "original_channels": channels,
                          "read_frame_count": frames, "selected_frame_count": 4 * rate, "start_frame": begin,
                          "start_s": begin / rate, "end_s": (begin + 4 * rate) / rate, "transform": _TRANSFORM,
                          "derived_wav_sha256": _digest(clip), "derived_frame_count": 4 * RATE}}
            record["descriptor_sha256"] = _digest(_canonical(record))
            with stage_manifest.open("x", encoding="utf-8") as target:
                json.dump(record, target, indent=2, ensure_ascii=False, allow_nan=False)
            # Same-directory hard links atomically create names and never replace
            # an existing file. The manifest is the final commit marker.
            if not wav.exists():
                os.link(stage_wav, wav)
            os.link(stage_manifest, manifest)
            # Only successful staging links are removed. Failed/partial bytes stay.
            for successful_stage in (stage_wav, stage_manifest):
                try:
                    successful_stage.unlink()
                except OSError:
                    # Committed data is usable; a remaining staging link appears
                    # in status warnings rather than falsifying import failure.
                    pass
            return self._load(identity)
