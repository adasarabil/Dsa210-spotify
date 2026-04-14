# DSA 210 – What Makes a Song Popular on Spotify?

**Student:** Ada Derviş Sarabil – 34362  
**Course:** DSA 210 Introduction to Data Science, Spring 2026

---

## Research Question

Does the emotional content of a song's lyrics — measured through NLP sentiment analysis — explain Spotify popularity beyond what audio features alone can capture? And when a song's musical tone and its lyrics point in opposite directions emotionally (e.g. upbeat melody but dark lyrics), does that tension itself predict higher popularity?

---

## Hypotheses

**H1:** Songs with positive lyrics sentiment have significantly higher Spotify popularity than songs with negative lyrics sentiment.  
- Test: Mann-Whitney U test (α = 0.05)

**H2:** Spotify's audio-derived valence is a weaker predictor of popularity than VADER lyrics sentiment score.  
- Test: Steiger's test for dependent correlations (α = 0.05)

**H3:** Songs with a high valence–sentiment gap (emotionally mismatched audio and lyrics) have higher popularity than low-gap songs.  
- Gap metric: `valence_sentiment_gap = |valence − normalized_sentiment|`  
- Test: Mann-Whitney U test (α = 0.05)

---

## Datasets

| Dataset | Source | Description |
|---|---|---|
| Spotify Songs with Lyrics | [Kaggle – imuhammad](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs) | ~18,000 tracks with audio features, lyrics, popularity scores and language tags |
| Spotify Tracks Dataset | [Kaggle – maharshipandya](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | ~114,000 tracks with audio features and popularity scores across 114 genres |

### How the datasets are used

The primary analysis runs on `spotify_songs.csv`, which already contains both audio features and lyrics in a single file — no external API needed. VADER sentiment analysis (`nltk.sentiment.vader`) is applied to the `lyrics` column to produce a `compound_sentiment` score (−1 to +1) per track. This score is independent of Spotify's own `valence` feature, which is derived from audio signals rather than text, making it a genuine external enrichment variable.

Two new features are engineered:
- `compound_sentiment` — lyrics-based emotional polarity from VADER
- `valence_sentiment_gap` — absolute difference between valence and normalized sentiment, capturing emotional mismatch between sound and lyrics

The larger `dataset.csv` (114K tracks) is used for genre-level and distribution comparisons where a broader sample is needed. Instrumental tracks (missing or very short lyrics) are flagged and analyzed as a subgroup.

---

## Repository Structure

```
your-repo/
├── data/
│   ├── spotify_songs.csv        ← primary (lyrics + audio features)
│   └── dataset.csv              ← secondary (broader genre coverage)
├── notebooks/
│   └── EDA_and_Hypothesis_Tests.ipynb
├── figures/
│   ├── popularity_distribution.png
│   ├── sentiment_distribution.png
│   ├── valence_vs_sentiment.png
│   ├── popularity_by_sentiment_category.png
│   ├── valence_sentiment_gap.png
│   └── correlation_heatmap.png
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
mkdir -p figures
jupyter notebook notebooks/EDA_and_Hypothesis_Tests.ipynb
```

Then run all cells (Cell → Run All).

---

## Requirements

See `requirements.txt`. Main dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `nltk`, `jupyter`.

---

## AI Usage Disclosure

This project used Claude (Anthropic) as an assistant for dataset selection, research question refinement, and code scaffolding, in accordance with DSA 210 guidelines. All analysis, interpretation, and final code were reviewed and validated by the student.
