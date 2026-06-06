"""
finetune_wav2vec2.py — fine-tune a pretrained anti-spoofing model on YOUR data.

Best of both worlds: starts from MelodyMachine/Deepfake-audio-detection-V2
(already trained on 900+ real + 900+ AI English voices, so it's robust on real
human speech) and adapts it to YOUR Hebrew AI + YOUR microphone using the files
in data/ai/ and data/human/.

Label convention (matches the base model's head):  0 = fake/AI,  1 = real/human.

Setup (in the venv):
    pip install torch transformers soundfile librosa

Data:
    data/ai/     -> AI / synthetic voice   (run gen_hebrew_ai.py to fill it)
    data/human/  -> real human voice
Run:
    python finetune_wav2vec2.py
Use the result:  put in .env ->  VOICE_ENGINE=pretrained
                                 SPOOF_MODEL=./wav2vec2_finetuned
"""
import os, glob, random, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

BASE_MODEL = os.environ.get("BASE_MODEL", "MelodyMachine/Deepfake-audio-detection-V2")
OUT_DIR    = "wav2vec2_finetuned"
SR         = 16000
CLIP       = SR * 4          # 4-second clips
EPOCHS     = 8
BATCH      = 4
LR         = 1e-5
HOLDOUT    = 0.15            # fraction held out per class for honest accuracy
SEED       = 42
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".aac")

random.seed(SEED); np.random.seed(SEED)


def gather_files():
    """Return list of (path, label). 0 = AI/fake, 1 = human/real."""
    items = []
    def add_dir(folder, label):
        for f in sorted(glob.glob(os.path.join(folder, "*"))):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                items.append((f, label))
    add_dir("data/ai", 0)
    add_dir("data/human", 1)
    # original labelled Hebrew files
    for p in ("text_fraud_recognition/hebrew_3_ai.mp3",
              "text_fraud_recognition/speach_hebrew_ai.mp3"):
        if os.path.isfile(p): items.append((p, 0))
    for f in sorted(glob.glob("text_fraud_recognition/28*.mp3")):
        items.append((f, 1))
    for p in ("text_fraud_recognition/test_call_human.m4a",
              "text_fraud_recognition/test_call_human_he.m4a"):
        if os.path.isfile(p): items.append((p, 1))
    # de-dup
    seen, out = set(), []
    for p, l in items:
        if p not in seen:
            seen.add(p); out.append((p, l))
    return out


def load_audio(path):
    try:
        import soundfile as sf
        w, sr = sf.read(path, dtype="float32", always_2d=False); sr = int(sr)
    except Exception:
        import librosa
        w, sr = librosa.load(path, sr=None, mono=True); sr = int(sr)
    w = np.asarray(w, np.float32)
    if w.ndim > 1: w = w.mean(axis=1)
    if sr != SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w


def crop_or_pad(w, train):
    if len(w) >= CLIP:
        start = random.randint(0, len(w) - CLIP) if train else (len(w) - CLIP) // 2
        return w[start:start + CLIP]
    return np.pad(w, (0, CLIP - len(w)))


def augment(w):
    if random.random() < 0.5:
        w = w + np.random.randn(len(w)).astype(np.float32) * random.uniform(0.001, 0.01)
    if random.random() < 0.5:
        w = w * random.uniform(0.7, 1.3)
    return np.clip(w, -1.0, 1.0)


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    torch.manual_seed(SEED)

    items = gather_files()
    n_ai = sum(1 for _, l in items if l == 0)
    n_hu = sum(1 for _, l in items if l == 1)
    print(f"Data: {len(items)} files  |  AI={n_ai}  human={n_hu}")
    if n_ai == 0 or n_hu == 0:
        print("\nNeed BOTH classes. Put AI files in data/ai/ (run gen_hebrew_ai.py) "
              "and human files in data/human/.")
        return

    # decode once (cache full waveforms)
    print("Decoding audio ...")
    cache, good = {}, []
    for p, l in items:
        try:
            cache[p] = load_audio(p); good.append((p, l))
        except Exception as e:
            print(f"  skip {os.path.basename(p)} ({type(e).__name__})")
    items = good

    # stratified holdout
    random.shuffle(items)
    by = {0: [x for x in items if x[1] == 0], 1: [x for x in items if x[1] == 1]}
    test, train = [], []
    for l, lst in by.items():
        k = max(1, int(len(lst) * HOLDOUT)) if len(lst) > 3 else 0
        test += lst[:k]; train += lst[k:]
    random.shuffle(train)
    print(f"Train={len(train)}  Holdout test={len(test)}")

    print(f"Loading base model {BASE_MODEL} ...")
    fe = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(BASE_MODEL)
    try:
        model.freeze_feature_encoder()   # keep low-level audio features fixed
    except Exception:
        pass
    model.train()

    # class weights (balance AI vs human)
    n0 = sum(1 for _, l in train if l == 0); n1 = len(train) - n0
    cw = torch.tensor([len(train) / (2 * max(n0, 1)),
                       len(train) / (2 * max(n1, 1))], dtype=torch.float32)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)

    def run_batch(batch, train_mode):
        wavs = [augment(crop_or_pad(cache[p], True)) if train_mode
                else crop_or_pad(cache[p], False) for p, _ in batch]
        labels = torch.tensor([l for _, l in batch], dtype=torch.long)
        inp = fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        logits = model(**inp).logits
        loss = F.cross_entropy(logits, labels, weight=cw)
        preds = logits.argmax(-1)
        return loss, (preds == labels).sum().item(), len(batch)

    for ep in range(1, EPOCHS + 1):
        random.shuffle(train)
        model.train()
        tot_loss = corr = seen = 0
        for i in range(0, len(train), BATCH):
            batch = train[i:i + BATCH]
            loss, c, n = run_batch(batch, True)
            loss.backward(); opt.step(); opt.zero_grad()
            tot_loss += loss.item() * n; corr += c; seen += n
        msg = f"epoch {ep}/{EPOCHS}  loss={tot_loss/seen:.4f}  train_acc={corr/seen:.2f}"
        # holdout eval
        if test:
            model.eval()
            with torch.no_grad():
                tc = 0
                for p, l in test:
                    _, c, _ = run_batch([(p, l)], False); tc += c
            msg += f"  holdout_acc={tc/len(test):.2f}"
        print(msg)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    fe.save_pretrained(OUT_DIR)
    print(f"\nSaved fine-tuned model -> {OUT_DIR}/")

    # final per-file report on the holdout
    if test:
        model.eval()
        print("\nHoldout files (AI% = P(fake)):")
        with torch.no_grad():
            for p, l in test:
                inp = fe([crop_or_pad(cache[p], False)], sampling_rate=SR, return_tensors="pt", padding=True)
                ai = float(torch.softmax(model(**inp).logits, -1)[0, 0].item()) * 100
                truth = "AI" if l == 0 else "human"
                print(f"  {os.path.basename(p):<38} truth={truth:<6} AI%={ai:5.1f}")

    print("\nTo use it, set in .env:\n  VOICE_ENGINE=pretrained\n  SPOOF_MODEL=./wav2vec2_finetuned")


if __name__ == "__main__":
    main()
