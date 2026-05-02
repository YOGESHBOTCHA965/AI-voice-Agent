"""Simple smoke test that avoids heavy audio/model deps.
Runs basic text utilities from `xtts_pipeline.py`.
"""
from xtts_pipeline import clean_text, split_text_for_two_segments

def run():
    s = "“It’s time: follow—now.” We’ll go."
    cleaned = clean_text(s)
    assert '"' in cleaned and 'cannot' not in cleaned  # basic sanity
    long = ' '.join(['word'] * 300)
    parts = split_text_for_two_segments(long)
    assert len(parts) == 2
    print('Smoke test passed: clean_text and split_text_for_two_segments')

if __name__ == '__main__':
    run()
