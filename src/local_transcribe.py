import os
import json
import whisper
import pandas as pd
from tqdm import tqdm

CSV_PATH = "data/dataset/csvs/train.csv"
AUDIO_DIR = "data/dataset/audios/train"
OUT_PATH  = "data/transcripts.csv"

df = pd.read_csv(CSV_PATH)
df['filename'] = df['filename'].astype(str).str.strip().apply(lambda x: x if x.endswith('.wav') else x + '.wav')

model = whisper.load_model("small")

rows = []
for fname in tqdm(df['filename']):
    path = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(path):
        rows.append((fname, ""))
        continue
    try:
        result = model.transcribe(path)
        text = result["text"]
    except:
        text = ""
    rows.append((fname, text))

pd.DataFrame(rows, columns=["filename","transcript"]).to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
