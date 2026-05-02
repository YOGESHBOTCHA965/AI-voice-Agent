"""
xtts_pipeline — small helper module extracted from notebooks.

Functions are written to avoid heavy imports at module import time; TTS and audio libs are imported inside functions.
"""
from typing import List

def clean_text(text: str) -> str:
    import re
    SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}
    for s, p in SMART_QUOTES.items():
        text = text.replace(s, p)
    CONTRACTIONS = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "it's": "it is", "I'm": "I am",
    }
    for c, e in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(c)}\b", e, text, flags=re.IGNORECASE)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*—\s*", " — ", text)
    text = re.sub(r"\s*([,])\s*", r", ", text)
    text = re.sub(r"\s*([\.\!\?])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_text_for_two_segments(text: str) -> List[str]:
    words = text.split()
    if len(words) < 50:
        return [text]
    SEGMENT_BREAK = 0.6
    cut = int(len(words) * SEGMENT_BREAK)
    window = min(len(words) - cut - 1, 80)
    for i in range(window):
        token = words[cut + i]
        if token.endswith('.') or token.endswith('!') or token.endswith('?'):
            cut = cut + i + 1
            break
    return [' '.join(words[:cut]), ' '.join(words[cut:])]

def cross_fade_join(wav1_path: str, wav2_path: str, out_path: str, fade_ms: int = 250) -> str:
    import numpy as np
    import soundfile as sf
    y1, sr1 = sf.read(wav1_path, always_2d=True)
    y2, sr2 = sf.read(wav2_path, always_2d=True)
    if sr1 != sr2:
        raise ValueError("Sample rates do not match")
    fade_samples = max(1, int(sr1 * (fade_ms / 1000.0)))
    if y1.shape[0] <= fade_samples or y2.shape[0] <= fade_samples:
        y = np.concatenate([y1, y2], axis=0)
        sf.write(out_path, y, sr1)
        return out_path
    a = y1[:-fade_samples]
    b = y2[fade_samples:]
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
    fade_in = 1.0 - fade_out
    overlap = y1[-fade_samples:] * fade_out + y2[:fade_samples] * fade_in
    y = np.concatenate([a, overlap, b], axis=0)
    sf.write(out_path, y, sr1)
    return out_path

def tts_single_shot(text: str, reference_wav: str, out_path: str):
    """Attempt single-shot XTTS inference using Coqui TTS API."""
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, speaker_wav=reference_wav, language='en', file_path=out_path)
    return out_path

def tts_two_segment_fallback(text: str, reference_wav: str, out_path: str):
    parts = split_text_for_two_segments(text)
    if len(parts) == 1:
        raise RuntimeError('Text too short for fallback split')
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    out_a = out_path.replace('.wav', '_A.wav')
    out_b = out_path.replace('.wav', '_B.wav')
    tts.tts_to_file(text=parts[0], speaker_wav=reference_wav, language='en', file_path=out_a)
    tts.tts_to_file(text=parts[1], speaker_wav=reference_wav, language='en', file_path=out_b)
    return cross_fade_join(out_a, out_b, out_path, fade_ms=300)
