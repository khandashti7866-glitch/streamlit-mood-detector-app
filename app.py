# app.py
"""
Mood Tracker & Analytics App (Streamlit)

Features:
- Log mood entries (auto date/time + text description)
- Automatic mood detection using TextBlob (if available) + keyword fallback
- Save entries to mood_log.csv (append)
- Analytics: mood frequency, daily trend, word cloud
- Export CSV
- No API keys required to run locally
"""

import os
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Try to import TextBlob; if not available, we'll fall back to a simple rule-based sentiment
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# Try to import nltk's SentimentIntensityAnalyzer if available (optional)
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except Exception:
    VADER_AVAILABLE = False

# ---------------------------
# Configuration & constants
# ---------------------------
DATA_FILE = "mood_log.csv"

# Mood categories and maps
MOOD_KEYWORDS = {
    "happy": ["happy", "joy", "glad", "good", "cheerful", "excited", "😊", "yay", "grateful"],
    "sad": ["sad", "depressed", "down", "unhappy", "😭", "miserable"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated", "😡"],
    "calm": ["calm", "relaxed", "chill", "peaceful", "serene"],
    "stressed": ["stressed", "anxious", "worried", "overwhelmed", "tense"],
    "energetic": ["energetic", "energetic", "energized", "motivated", "pumped"],
    "neutral": ["okay", "fine", "neutral", "so-so"]
}

# Simple numeric mapping for trend / average mood score
MOOD_SCORE = {
    "angry": -2,
    "sad": -1,
    "neutral": 0,
    "calm": 1,
    "energetic": 2,
    "stressed": -1,
    "happy": 2
}

MOOD_EMOJI = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😡",
    "calm": "😌",
    "stressed": "😰",
    "energetic": "⚡️",
    "neutral": "😐"
}

MOOD_COLOR = {
    "happy": "#2ECC71",     # green
    "sad": "#3498DB",       # blue
    "angry": "#E74C3C",     # red
    "calm": "#1ABC9C",      # turquoise
    "stressed": "#F1C40F",  # yellow
    "energetic": "#9B59B6", # purple
    "neutral": "#95A5A6"    # gray
}

# ---------------------------
# Utilities
# ---------------------------
def ensure_data_file(path=DATA_FILE):
    """Create CSV with headers if not exists."""
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["timestamp", "date", "time", "description", "mood", "mood_score"])
        df.to_csv(path, index=False)

def load_data(path=DATA_FILE):
    ensure_data_file(path)
    return pd.read_csv(path, parse_dates=["timestamp"], dayfirst=False)

def save_entry(description, mood, mood_score, path=DATA_FILE):
    ensure_data_file(path)
    now = datetime.now()
    row = {
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.time().strftime("%H:%M:%S"),
        "description": description,
        "mood": mood,
        "mood_score": mood_score
    }
    df = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not os.path.exists(path) or os.path.getsize(path) == 0, index=False)

def keyword_lookup(text):
    """Return mood based on keyword matching. Returns None if not matched."""
    text_lower = text.lower()
    hits = defaultdict(int)
    for mood, keywords in MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                hits[mood] += 1
    if not hits:
        return None
    # choose mood with highest hits; ties broken arbitrarily by sorted order
    chosen = max(hits.items(), key=lambda x: (x[1], x[0]))[0]
    return chosen

def polarity_to_mood(polarity, subjectivity=None):
    """Map numeric polarity (-1..1) to mood label."""
    # thresholds can be tuned
    if polarity >= 0.5:
        return "happy"
    elif 0.1 <= polarity < 0.5:
        return "calm"
    elif -0.1 < polarity < 0.1:
        return "neutral"
    elif -0.5 < polarity <= -0.1:
        return "sad"
    else:
        return "angry"

def detect_mood(text):
    """Detect mood using TextBlob, VADER if available; fallback to keyword mapping."""
    text = text.strip()
    if not text:
        return "neutral", MOOD_SCORE["neutral"]

    # Keyword-based first pass (strong signal)
    kw = keyword_lookup(text)
    if kw:
        return kw, MOOD_SCORE.get(kw, 0)

    # Try TextBlob polarity
    if TEXTBLOB_AVAILABLE:
        try:
            tb = TextBlob(text)
            polarity = tb.sentiment.polarity  # -1..1
            mood = polarity_to_mood(polarity, subjectivity=tb.sentiment.subjectivity)
            return mood, MOOD_SCORE.get(mood, 0)
        except Exception:
            pass

    # Try VADER if available
    if VADER_AVAILABLE:
        try:
            scores = _vader.polarity_scores(text)
            polarity = scores["compound"]  # -1..1
            mood = polarity_to_mood(polarity)
            return mood, MOOD_SCORE.get(mood, 0)
        except Exception:
            pass

    # Final fallback: neutral
    return "neutral", MOOD_SCORE["neutral"]

# ---------------------------
# Analytics helpers
# ---------------------------
def mood_frequency_chart(df):
    freq = df["mood"].value_counts().reset_index()
    freq.columns = ["mood", "count"]
    fig = px.bar(freq, x="mood", y="count", color="mood",
                 color_discrete_map=MOOD_COLOR,
                 title="Mood Frequency")
    fig.update_layout(showlegend=False)
    return fig

def daily_trend(df):
    # Ensure timestamp parsed
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["date_only"] = df["timestamp"].dt.date
    # average mood score per day
    trend = df.groupby("date_only")["mood_score"].mean().reset_index()
    trend = trend.sort_values("date_only")
    fig = px.line(trend, x="date_only", y="mood_score", title="Daily Average Mood Score",
                  markers=True)
    fig.update_yaxes(title="Average mood score (higher = better)")
    return fig, trend

def happiest_day(df):
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date_only"] = df["timestamp"].dt.date
    agg = df.groupby("date_only")["mood_score"].mean()
    best_day = agg.idxmax()
    best_score = agg.max()
    return str(best_day), float(best_score)

def most_frequent_mood(df):
    if df.empty:
        return None
    return df["mood"].mode().iat[0]

def average_mood_score(df):
    if df.empty:
        return 0.0
    return float(df["mood_score"].mean())

def generate_wordcloud(text_series, max_words=100):
    text = " ".join([str(x) for x in text_series.dropna().values])
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words)
    wc = wc.generate(text)
    return wc

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Mood Tracker", layout="wide", initial_sidebar_state="expanded")

st.title("Mood Tracker & Analytics")
st.markdown("Record short mood descriptions 3–4 times a day. The app detects mood from text and shows analytics.")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Log Mood", "View Analytics", "Settings"])

# Settings: show info about detectors
if page == "Settings":
    st.subheader("Settings & Diagnostics")
    st.write("TextBlob available:", TEXTBLOB_AVAILABLE)
    st.write("VADER available:", VADER_AVAILABLE)
    st.write("Data file:", DATA_FILE)
    st.write("Mood keywords (sample):")
    for m, ks in MOOD_KEYWORDS.items():
        st.write(f"- {m}: {', '.join(ks[:8])}")
    st.markdown("---")
    st.write("You can export the CSV from the Analytics page.")
    st.stop()

# Ensure data file exists
ensure_data_file(DATA_FILE)

# ---------------------------
# Log Mood page
# ---------------------------
if page == "Log Mood":
    st.header("Log Mood")
    st.write("Write a short description of how you're feeling. The app will detect your mood from the text.")
    with st.form("mood_form"):
        desc = st.text_area("How are you feeling? (a few words or a sentence)", height=120)
        # optionally let user pick a time of day tag (not required)
        time_of_day = st.selectbox("Time (optional)", ["Auto (now)", "Morning", "Afternoon", "Evening", "Night"])
        submitted = st.form_submit_button("Save Mood")
    if submitted:
        mood_label, mood_score = detect_mood(desc)
        save_entry(desc, mood_label, mood_score)
        emoji = MOOD_EMOJI.get(mood_label, "")
        st.success(f"Saved — Detected mood: **{mood_label} {emoji}** (score: {mood_score})")
        st.write("You can view analytics in the sidebar → View Analytics")

# ---------------------------
# View Analytics page
# ---------------------------
if page == "View Analytics":
    st.header("Analytics")
    df = load_data(DATA_FILE)

    if df.empty or len(df) == 0:
        st.info("No mood entries yet. Go to 'Log Mood' to add your first entry.")
        st.stop()

    # Date filter
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    min_date = df["timestamp"].dt.date.min()
    max_date = df["timestamp"].dt.date.max()
    st.sidebar.subheader("Filters")
    start_date = st.sidebar.date_input("Start date", min_value=min_date, value=min_date)
    end_date = st.sidebar.date_input("End date", min_value=min_date, value=max_date)
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mf = most_frequent_mood(df_filtered)
        st.metric("Most frequent mood", f"{mf} {MOOD_EMOJI.get(mf, '')}" if mf else "—")
    with col2:
        hd = happiest_day(df_filtered)
        if hd:
            day_str, score = hd
            st.metric("Happiest day", day_str, f"{score:.2f}")
        else:
            st.metric("Happiest day", "—")
    with col3:
        avg = average_mood_score(df_filtered)
        st.metric("Average mood score", f"{avg:.2f}")
    with col4:
        total = len(df_filtered)
        st.metric("Entries in range", total)

    st.markdown("---")

    # Charts
    st.subheader("Mood Frequency")
    fig_freq = mood_frequency_chart(df_filtered)
    st.plotly_chart(fig_freq, use_container_width=True)

    st.subheader("Daily Mood Trend")
    fig_trend, trend_df = daily_trend(df_filtered)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Word Cloud of Descriptions")
    wc = generate_wordcloud(df_filtered["description"])
    if wc is None:
        st.info("No text available to generate word cloud.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Raw entries (preview)")
    show_cols = ["timestamp", "date", "time", "description", "mood", "mood_score"]
    st.dataframe(df_filtered[show_cols].sort_values("timestamp", ascending=False).reset_index(drop=True))

    # CSV export
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Export filtered CSV", csv, file_name=f"mood_log_{start_date}_to_{end_date}.csv", mime="text/csv")

    st.markdown("Tip: To back up all data, download the full `mood_log.csv` file from the project folder.")

# ---------------------------
# Footer / credits
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ — Mood Tracker")
st.sidebar.markdown("You can edit detection logic in `detect_mood()` and mood maps near the top of `app.py`.")
