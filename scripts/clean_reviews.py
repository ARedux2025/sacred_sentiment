from pathlib import Path
import pandas as pd
import re
from nltk.corpus import stopwords

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# Stopwords
STOP = set(stopwords.words("english"))

def basic_clean(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces, remove stopwords."""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)   # punctuation → space
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if w not in STOP)

def main():
    df = pd.read_csv(RAW / "reviews.csv")
    df = df.rename(columns={"place": "location", "text": "review_text"})
    df = df.drop_duplicates(subset=["review_text"]).copy()
    df["clean_text"] = df["review_text"].apply(basic_clean)
    out = CLEAN / "reviews_clean.csv"
    df.to_csv(out, index=False)
    print("Saved:", out.resolve())

if __name__ == "__main__":
    
    print("🚀 Script started")
    main()


import nltk
nltk.download('stopwords')

from pathlib import Path
import pandas as pd
import re

# Works no matter where you run it from
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews.csv"
OUT_DIR = ROOT / "data" / "clean"
OUT = OUT_DIR / "reviews_clean.csv"

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"http\S+|www\.\S+", "", s)  # remove URLs
    s = re.sub(r"\s+", " ", s)              # collapse whitespace
    return s

def main():
    print("📝 Script started")
    print("Reading:", RAW)
    if not RAW.exists():
        raise FileNotFoundError(f"Cannot find: {RAW}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(RAW)
    except UnicodeDecodeError:
        df = pd.read_csv(RAW, encoding="utf-8-sig")

    # Choose a text column
    text_col = next((c for c in ["text","review","review_text","content","comment","body"] if c in df.columns), None)
    if text_col is None:
        text_col = next((c for c in df.columns if df[c].dtype == "object"), None)
        if text_col is None:
            raise ValueError("No text column found to clean.")

    before = len(df)
    df[text_col] = df[text_col].map(clean_text)
    df = df[df[text_col].str.len() > 0].drop_duplicates()

    df.to_csv(OUT, index=False)
    print(f"✅ Saved: {OUT.resolve()} (rows: {len(df)}/{before})")

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd
import re

# Works no matter where you run it from
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews.csv"
OUT_DIR = ROOT / "data" / "clean"
OUT = OUT_DIR / "reviews_clean.csv"

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"http\S+|www\.\S+", "", s)  # remove URLs
    s = re.sub(r"\s+", " ", s)              # collapse whitespace
    return s

def main():
    print("📝 Script started")
    print("Reading:", RAW)
    if not RAW.exists():
        raise FileNotFoundError(f"Cannot find: {RAW}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(RAW)
    except UnicodeDecodeError:
        df = pd.read_csv(RAW, encoding="utf-8-sig")

    # Choose a text column
    text_col = next((c for c in ["text","review","review_text","content","comment","body"] if c in df.columns), None)
    if text_col is None:
        text_col = next((c for c in df.columns if df[c].dtype == "object"), None)
        if text_col is None:
            raise ValueError("No text column found to clean.")

    before = len(df)
    df[text_col] = df[text_col].map(clean_text)
    df = df[df[text_col].str.len() > 0].drop_duplicates()

    df.to_csv(OUT, index=False)
    print(f"✅ Saved: {OUT.resolve()} (rows: {len(df)}/{before})")

if __name__ == "__main__":
    main()

    print("RAW:", RAW.resolve(), "exists:", RAW.exists())
print("OUT:", OUT.resolve())
