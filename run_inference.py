"""
Lightweight demo runner for the XTTS long-form notebook.
Usage:
  python run_inference.py --reference reference.wav --out demo_output.wav --script-file myscript.txt

This script tries a single-shot generation first and falls back to a two-segment generate+crossfade on error.
"""
import argparse
import os
from pathlib import Path

def clean_text(text: str) -> str:
    import re
    SMART_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'"}
    for s, p in SMART_QUOTES.items():
        text = text.replace(s, p)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*—\s*", " — ", text)
    text = re.sub(r"\s*([,])\s*", r", ", text)
    text = re.sub(r"\s*([\.\!\?])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cross_fade_join(wav1_path: str, wav2_path: str, out_path: str, fade_ms: int = 250):
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

def split_text_for_two_segments(text: str):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', '-r', required=True)
    parser.add_argument('--out', '-o', default='demo_output.wav')
    parser.add_argument('--script-file', '-s', help='Text file with script (utf-8). If omitted, a short sample is used.')
    args = parser.parse_args()

    if not Path(args.reference).exists():
        raise FileNotFoundError(f"Missing reference WAV: {args.reference}")

    if args.script_file:
        script = Path(args.script_file).read_text(encoding='utf-8')
    else:
        script = (
            "This is a short demo of the long form TTS pipeline. "
            "Replace with your own script or use --script-file to load a longer script."
        )

    text = clean_text(script)

    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print('Attempting single-shot generation...')
        tts.tts_to_file(text=text, speaker_wav=args.reference, language='en', file_path=args.out)
        print('Wrote:', args.out)
        return
    except Exception as e:
        print('Single-shot failed, falling back to two-segment flow:', e)

    # Fallback: split, generate two parts, then crossfade
    parts = split_text_for_two_segments(text)
    if len(parts) == 1:
        raise RuntimeError('Fallback attempted but could not split text.')
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    out_a = 'demo_segA.wav'
    out_b = 'demo_segB.wav'
    print('Generating segment A...')
    tts.tts_to_file(text=parts[0], speaker_wav=args.reference, language='en', file_path=out_a)
    print('Generating segment B...')
    tts.tts_to_file(text=parts[1], speaker_wav=args.reference, language='en', file_path=out_b)
    print('Crossfading and writing final output...')
    cross_fade_join(out_a, out_b, args.out, fade_ms=300)
    print('Wrote:', args.out)

if __name__ == '__main__':
    main()
