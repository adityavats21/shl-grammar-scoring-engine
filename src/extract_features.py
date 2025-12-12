# src/extract_features.py
import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

TRAIN_CSV = "data/train.csv"
AUDIO_DIR = "data/train_audio"
TRANSCRIPTS = "data/transcripts.csv"  # created by local_transcribe.py
OUT_FEATURES = "data/features_audio.csv"
OUT_MERGED = "data/features_merged.csv"

df = pd.read_csv(TRAIN_CSV)
df['filename'] = df['filename'].astype(str).str.strip().apply(lambda x: x if x.lower().endswith('.wav') else x + '.wav')

def extract(path):
    y, sr = librosa.load(path, sr=16000)
    dur = len(y)/sr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    sc = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    sbw = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    roll = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except:
        tempo = np.nan
    out = {
        "duration": dur, "zcr": zcr, "rms": rms,
        "spec_centroid": sc, "spec_bandwidth": sbw, "rolloff": roll, "tempo": tempo
    }
    for i in range(13):
        out[f"mfcc_{i+1}_mean"] = float(mfcc_mean[i])
        out[f"mfcc_{i+1}_std"] = float(mfcc_std[i])
    return out

rows = []
for fname in tqdm(df['filename'].values, desc="Extracting audio features"):
    path = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(path):
        rows.append({"filename": fname})
        continue
    feats = extract(path)
    feats["filename"] = fname
    rows.append(feats)

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUT_FEATURES, index=False)
print("Saved audio features:", OUT_FEATURES)

if os.path.exists(TRANSCRIPTS):
    trans = pd.read_csv(TRANSCRIPTS)
    trans['filename'] = trans['filename'].astype(str).str.strip().apply(lambda x: x if x.lower().endswith('.wav') else x + '.wav')
    merged = df.merge(feat_df, on="filename", how="left").merge(trans, on="filename", how="left")
else:
    merged = df.merge(feat_df, on="filename", how="left")

merged.to_csv(OUT_MERGED, index=False)
print("Saved merged features:", OUT_MERGED)
