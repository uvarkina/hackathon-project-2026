"""
gen_hebrew_ai.py — generate FRESH Hebrew AI voice samples with gTTS (Google TTS).

These are NOT in the training set, so they give an HONEST test of whether the
model detects Hebrew synthetic speech (run check_model.py on the output folder).

One-time:
    pip install gTTS

Run (inside the venv):
    python gen_hebrew_ai.py

Output: fresh_test_hebrew/ai/*.mp3
Then:   python check_model.py fresh_test_hebrew/ai
"""
import os, json, time, sys

OUT_DIR = "data/ai"   # straight into the training set
PHRASES_JSON = "text_fraud_recognition/fraud_phrases.json"
N_SAMPLES = 40         # use up to this many distinct sentences
MIN_LEN = 20           # only full sentences, not single keywords
LANG = "iw"            # gTTS Hebrew language code ('iw'; some versions also accept 'he')
SLOW_VARIANTS = True   # also generate a slow version of each -> 2x samples, more variety


def load_phrases():
    with open(PHRASES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    he = data.get("he", [])
    # keep real sentences, drop short keywords, de-dup, preserve order
    seen, out = set(), []
    for p in he:
        p = p.strip()
        if len(p) >= MIN_LEN and p not in seen:
            seen.add(p); out.append(p)
    return out[:N_SAMPLES]


def main():
    try:
        from gtts import gTTS
    except ImportError:
        print("gTTS not installed. Run:  pip install gTTS")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    phrases = load_phrases()
    if not phrases:
        print("No suitable Hebrew sentences found in", PHRASES_JSON)
        sys.exit(1)

    # (text, speed_tag, slow_flag) jobs
    jobs = [(t, "n", False) for t in phrases]
    if SLOW_VARIANTS:
        jobs += [(t, "s", True) for t in phrases]

    print(f"Generating {len(jobs)} Hebrew AI samples into {OUT_DIR}/ ...\n")
    made = 0
    for i, (text, tag, slow) in enumerate(jobs, 1):
        path = os.path.join(OUT_DIR, f"gtts_he_{tag}_{i:03d}.mp3")
        try:
            gTTS(text=text, lang=LANG, slow=slow).save(path)
            made += 1
            print(f"  [{i:03d}] {os.path.basename(path)}  «{text[:38]}…»")
            time.sleep(0.4)  # be gentle on the endpoint
        except Exception as e:
            print(f"  [{i:03d}] FAILED ({type(e).__name__}: {e})")
            if "iw" in LANG:
                print("       tip: try editing LANG = 'he' at the top of this file")
            break

    print(f"\nDone — {made} AI files in {OUT_DIR}/.")
    print("Now retrain:  python retrain_model.py")


if __name__ == "__main__":
    main()
