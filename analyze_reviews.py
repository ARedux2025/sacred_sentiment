from pathlib import Path
import matplotlib.pyplot as plt

print("RUNNING FILE:", Path(__file__).resolve())

# Detect project root whether this file is at ...\sacred_sentiment\ or ...\sacred_sentiment\scripts\
ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").exists():
    ROOT = ROOT.parent  # handles the case when this file is inside scripts/

CSV_PATH  = ROOT / "data" / "raw"   / "reviews.csv"
CLEAN_PATH = ROOT / "data" / "clean" / "reviews_clean.csv"
OUT_DIR   = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT:", ROOT)
print("Reading:", CSV_PATH.resolve(), "exists:", CSV_PATH.exists())

# Fallback to clean file if raw isn't there
if not CSV_PATH.exists() and CLEAN_PATH.exists():
    print("Raw file not found, falling back to:", CLEAN_PATH)
    CSV_PATH = CLEAN_PATH


# If neither exists, stop with a clear error
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Could not find reviews at:\n  {ROOT / 'data' / 'raw' / 'reviews.csv'}\n  or\n  {ROOT / 'data' / 'clean' / 'reviews_clean.csv'}"
    )

# ---- Analysis ----
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Load CSV (safe encoding fallback)
try:
    df = pd.read_csv(CSV_PATH)
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # ✅ Step 2: Fix place names
df['place'] = df['place'].str.strip().str.title()   # normalize spacing/case

fixes = {
    "African Burial Grounds Nationa": "African Burial Grounds National Monument",
    "St. Patricks Cathedral": "Saint Patricks Cathedral",
    "Islamic Cultural Ctr Of New Yor": "Islamic Cultural Center Of New York"
}
df['place'] = df['place'].replace(fixes)

# --- Fix known typos in place names ---
fixes = {
    "African Burial Grounds National Monumnet": "African Burial Grounds National Monument",
    "Islamic Cultual Center Of New York": "Islamic Cultural Center Of New York",
    "Saint Patricks Cathdral": "Saint Patricks Cathedral",
}

# Apply replacements
df['place'] = df['place'].replace(fixes)


# Step 3: Save cleaned dataset
clean_path = ROOT / "data" / "clean" / "reviews_places_fixed.csv"
df.to_csv(clean_path, index=False)
print("Cleaned reviews saved to:", clean_path.resolve())
print("Reviews loaded:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())
location_counts = df['place'].value_counts()
print("\nReviews per location:")
print(location_counts)

# Step 4: Print each location neatly
for loc, count in location_counts.items():
    print(f"{loc}: {count} reviews")



# Pick a text column automatically
TEXT_CANDIDATES = ["review", "text", "review_text", "content", "comment", "body"]
TEXT_COL = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
if TEXT_COL is None:
    TEXT_COL = next((c for c in df.columns if df[c].dtype == "object"), None)
if TEXT_COL is None:
    raise ValueError("No text-like column found to analyze.")
print("Text column:", TEXT_COL)

# VADER sentiment
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
df["vader_compound"] = df[TEXT_COL].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
def label(c):
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
df["vader_label"] = df["vader_compound"].apply(label)

# Print a small preview
print("\nRESULTS (first few)")
print(df[[TEXT_COL, "vader_compound", "vader_label"]].head().to_string(index=False))
print("Counts:", df["vader_label"].value_counts().to_dict())

# ---- Save to data/processed (handles 'file open in Excel') ----
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_reviews = OUT_DIR / "reviews_with_sentiment.csv"
print("\nWriting to:", out_reviews.resolve())
try:
    df.to_csv(out_reviews, index=False)
except PermissionError:
    stamped = OUT_DIR / f"reviews_with_sentiment_{datetime.now():%Y%m%d_%H%M%S}.csv"
    print("Output was locked; saved as:", stamped.name)
    df.to_csv(stamped, index=False)
    out_reviews = stamped

print("Saved file:", out_reviews.resolve())
# --- Extras: sentiment bar chart + top terms ---
import matplotlib.pyplot as plt
import re
from collections import Counter

# 1) Bar chart of sentiment counts
counts = df["vader_label"].value_counts().reindex(["negative","neutral","positive"]).fillna(0).astype(int)
plt.figure()
counts.plot(kind="bar")
plt.title("Sentiment counts")
plt.tight_layout()
out_png = OUT_DIR / "sentiment_counts.png"
plt.savefig(out_png)
plt.close()
print("Saved chart:", out_png.resolve())

# --- New: plot number of reviews per location ---
plt.figure(figsize=(8,5))
location_counts.plot(kind="bar")
plt.title("Number of Reviews per Location")
plt.ylabel("Review Count")
plt.xlabel("Location")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "location_review_counts.png")
plt.close()


# 2) Simple top terms (no extra downloads)
STOP = {
    "a","an","the","and","or","to","of","in","on","for","with","is","are","was","were","be","been","being",
    "this","that","those","these","it","its","it's","i","im","i'm","ive","i've","we","we're","you","your",
    "they","their","them","at","by","from","as","but","so","if","than","then","there","here","very","really"
}
tokens = []
for text in df[TEXT_COL].astype(str):
    words = re.findall(r"[A-Za-z']+", text.lower())
    tokens.extend(w for w in words if w not in STOP and len(w) > 2)

top = Counter(tokens).most_common(25)
pd.DataFrame(top, columns=["term","count"]).to_csv(OUT_DIR / "top_terms.csv", index=False)
print("Saved terms:", (OUT_DIR / "top_terms.csv").resolve())

# --- Word cloud(s) ---
try:
    from wordcloud import WordCloud

    from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Step: Stronger filter for negative reviews
negative_reviews = df[df["vader_compound"] <= -0.3]  # only strong negatives

# Add extra stopwords (remove non-informative words)
extra_stops = {"visit", "place", "temple", "behavior", "anyone", "new", "york"}
STOP.update(extra_stops)

# Join all negative review text
neg_text = " ".join(negative_reviews[TEXT_COL].dropna().astype(str))

# Generate word cloud
wc_neg = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=STOP,
    collocations=False
).generate(neg_text)

# Save and show
out_png = OUT_DIR / "wordcloud_negative.png"
wc_neg.to_file(out_png)
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.show()

print("Saved cleaned negative word cloud to:", out_png.resolve())

# --- Strong negative word cloud (only clearly negative words) ---






    # overall cloud using the tokens list you already built
    text_all = " ".join(tokens)
    if text_all:
        wc_all = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(text_all)
        out_wc_all = OUT_DIR / "wordcloud_all.png"
        wc_all.to_file(str(out_wc_all))
        print("Saved word cloud:", out_wc_all.resolve())
        # === Clean negative word cloud (filter out weak words) ===
NEG_FILTER = {"visit", "place", "people", "history", "new", "york", "temple"}


    # helper to make clouds from a Series of text
    def make_wc(rows, filename):
        txt = " ".join(
            w for t in rows.astype(str)
            for w in re.findall(r"[A-Za-z']+", t.lower())
            if w not in STOP and len(w) > 2
        )
        if txt:
            wc = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(txt)
            path = OUT_DIR / filename
            wc.to_file(str(path))
            print("Saved word cloud:", path.resolve())

    # per-sentiment clouds (optional)
    make_wc(df.loc[df["vader_label"] == "positive", TEXT_COL], "wordcloud_positive.png")
    make_wc(df.loc[df["vader_label"] == "negative", TEXT_COL], "wordcloud_negative.png")

except Exception as e:
    print("Skipping word clouds (module missing or other issue):", e)

# --- Sentiment-only words (Hu & Liu opinion lexicon) ---
import nltk
nltk.download('opinion_lexicon', quiet=True)
from nltk.corpus import opinion_lexicon

pos_set = set(opinion_lexicon.positive())
neg_set = set(opinion_lexicon.negative())

sentiment_tokens = [w for w in tokens if w in pos_set or w in neg_set]

# Top sentiment words
from collections import Counter
sentiment_counts = Counter(sentiment_tokens).most_common(50)
pd.DataFrame(sentiment_counts, columns=["term","count"]).to_csv(OUT_DIR / "sentiment_terms_top.csv", index=False)
print("Saved:", (OUT_DIR / "sentiment_terms_top.csv").resolve())

# Word cloud limited to sentiment words
try:
    from wordcloud import WordCloud
    if sentiment_tokens:
        wc_sent = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(" ".join(sentiment_tokens))
        (OUT_DIR / "wordcloud_sentiment_only.png").write_bytes(wc_sent.to_image().tobytes())  # fallback if .to_file fails on some setups
        wc_sent.to_file(str(OUT_DIR / "wordcloud_sentiment_only.png"))  # normal path
        print("Saved:", (OUT_DIR / "wordcloud_sentiment_only.png").resolve())
except Exception as e:
    print("Skipping sentiment-only wordcloud:", e)



# --- By-location summary (runs only if a 'location' column exists) ---
import re
import matplotlib.pyplot as plt  # harmless if already imported

if "location" in df.columns:
    # counts and percentages by location
    counts = df.groupby(["location", "vader_label"]).size().unstack(fill_value=0)
    # ensure consistent column order
    for col in ["negative", "neutral", "positive"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["negative", "neutral", "positive"]]
    perc = counts.div(counts.sum(axis=1), axis=0).round(3)

    # save tables
    counts.to_csv(OUT_DIR / "location_sentiment_counts.csv")
    perc.to_csv(OUT_DIR / "location_sentiment_percent.csv")
    print("Saved:", (OUT_DIR / "location_sentiment_counts.csv").resolve())
    print("Saved:", (OUT_DIR / "location_sentiment_percent.csv").resolve())

    # safe filenames for charts
    def slug(s):
        return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

    # bar chart per location
    for loc, row in counts.iterrows():
        plt.figure()
        row.plot(kind="bar")
        plt.title(f"Sentiment at {loc}")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sentiment_{slug(loc)}.png")
        plt.close()

# A) Reviews per location (clean table)
location_counts = df['place'].value_counts()
location_counts.to_csv(OUT_DIR / "location_review_counts.csv", header=["review_count"])
print("Saved:", (OUT_DIR / "location_review_counts.csv").resolve())

# B) Sentiment breakdown per location (clean table)
sentiment_counts = (
    df.groupby(['place', 'vader_label'])
      .size()
      .unstack(fill_value=0)              # columns: negative / neutral / positive
      .reindex(location_counts.index)     # same order as counts
)
sentiment_counts.to_csv(OUT_DIR / "location_sentiment_counts.csv")
print("Saved:", (OUT_DIR / "location_sentiment_counts.csv").resolve())

# --- Build a compact per-location summary ---
counts = pd.read_csv(OUT_DIR / "location_review_counts.csv", index_col=0).rename(columns={"review_count":"n_reviews"})
sent = pd.read_csv(OUT_DIR / "location_sentiment_counts.csv", index_col=0)

# make sure all columns exist
for col in ["negative","neutral","positive"]:
    if col not in sent.columns: sent[col] = 0

summary = (
    counts.join(sent[["negative","neutral","positive"]])
          .fillna(0)
          .astype({"negative":"int","neutral":"int","positive":"int"})
)

# positivity rate (guard against divide-by-zero)
total = summary[["negative","neutral","positive"]].sum(axis=1).replace(0,1)
summary["positive_rate"] = (summary["positive"] / total).round(3)

out_summary = OUT_DIR / "location_summary.csv"
summary.sort_values(["positive_rate","n_reviews"], ascending=[False, False]).to_csv(out_summary)
print("Saved:", out_summary.resolve())

 # --- (Optional) one chart: positivity per location ---
plt.figure(figsize=(12, 6))
summary.sort_values("positive_rate", ascending=False)["positive_rate"].plot(kind="bar")
plt.title("Share of Positive Reviews by Location")
plt.ylabel("Positive Rate")
plt.xlabel("Location")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
out_png = OUT_DIR / "location_positive_rate.png"
plt.savefig(out_png)
plt.close()
print("Saved chart:", out_png.resolve())
   

print("DONE")






