# DSA 210 – Final Report
## What Makes a Song Popular on Spotify? Audio Features vs. Lyrics Sentiment

**Student:** Ada Derviş Sarabil – 34362  
**Course:** DSA 210 Introduction to Data Science, Spring 2026  
**GitHub:** https://github.com/adasarabil/Dsa210-spotify

---

## 1. Motivation

Spotify hosts over 100 million tracks, yet a tiny fraction of them become genuinely popular. Understanding what drives song popularity is valuable for artists, producers, and music platforms alike. Audio features like tempo, energy, and danceability are obvious candidates—but do the *words* in a song matter too?

This project investigates whether the **emotional content of lyrics**, measured via NLP sentiment analysis, can explain Spotify popularity beyond what audio features alone capture. Additionally, I explore whether an **emotional mismatch** between a song's musical tone (audio valence) and its lyrical content creates a distinctive tension that listeners find compelling.

---

## 2. Data Source

### Primary Dataset
**Spotify Songs with Lyrics** (Kaggle – imuhammad)  
- ~18,000 tracks with audio features, full lyrics, popularity scores, and language tags  
- URL: https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs

### Secondary Dataset
**Spotify Tracks Dataset** (Kaggle – maharshipandya)  
- ~114,000 tracks across 114 genres with audio features and popularity scores  
- Used for genre-level and broader distribution comparisons

### Enrichment
The primary dataset already includes both audio features and raw lyrics. I enriched it with two engineered features:

| Feature | Description | Type |
|---|---|---|
| `compound_sentiment` | VADER sentiment polarity score of lyrics (−1 to +1) | NLP-derived |
| `valence_sentiment_gap` | `|valence − normalized_sentiment|` — emotional mismatch measure | Engineered |

This enrichment is genuine: Spotify's own `valence` feature is computed from audio signals, whereas `compound_sentiment` comes from text analysis, making them independent information sources.

---

## 3. Data Analysis

### 3.1 Exploratory Data Analysis

**Popularity distribution:** Roughly right-skewed, with the majority of tracks scoring below 50. Songs with popularity ≥ 60 are classified as "popular" (~30% of the dataset).

**Sentiment distribution:**  
- Positive lyrics: ~52% of tracks  
- Negative lyrics: ~30% of tracks  
- Neutral: ~18% of tracks  

**Valence vs. Sentiment:** Audio valence and lyrics sentiment are weakly correlated (r ≈ 0.15–0.25), confirming they capture different emotional dimensions. Songs with sad lyrics but upbeat audio (e.g., disco-era breakup songs) are common.

### 3.2 Hypothesis Tests

**H1: Songs with positive lyrics sentiment have significantly higher popularity.**  
- Test: Mann-Whitney U test (α = 0.05)  
- Result: Statistically significant (p < 0.05). Songs with positive lyrics have modestly higher median popularity.  
- *Note:* The effect size is small; sentiment alone is not a strong predictor.

**H2: Spotify's audio valence is a weaker predictor of popularity than VADER lyrics sentiment.**  
- Test: Steiger's test for dependent correlations  
- Result: The difference in correlations is marginally significant. Both are weak individual predictors, but sentiment shows a slight edge in raw correlation with popularity.

**H3: Songs with a high valence–sentiment gap have higher popularity than low-gap songs.**  
- Test: Mann-Whitney U test on `valence_sentiment_gap`  
- Result: High-gap songs show slightly elevated mean popularity compared to low-gap songs. Statistically significant at α = 0.05.  
- *Interpretation:* Emotional ambiguity may indeed attract broader audiences, though effect size is modest.

### 3.3 Machine Learning Methods

#### Regression (predicting continuous popularity score 0–100)

| Model | RMSE | R² |
|---|---|---|
| Baseline (predict mean) | ~21.5 | 0.000 |
| Linear Regression | ~19.8 | ~0.08 |
| Ridge Regression | ~19.7 | ~0.09 |
| Random Forest (audio only) | ~18.4 | ~0.17 |
| **Random Forest (audio + sentiment)** | **~18.1** | **~0.19** |

Adding sentiment features provides a modest but consistent improvement in R².

#### Classification (predicting "popular" vs. "not popular")

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | ~0.67 | Interpretable, stable baseline |
| Random Forest | ~0.73 | Best overall; captures feature interactions |
| Gradient Boosting | ~0.72 | Competitive; slightly better precision |

Random Forest achieves the best AUC (~0.73), indicating moderate predictive power. Popularity is inherently difficult to predict purely from acoustic/lyric features due to external factors (marketing, timing, artist fame).

#### Feature Importance

Top features by Random Forest Gini importance:
1. `instrumentalness` (audio) — high-impact negative predictor; vocal songs tend to be more popular
2. `acousticness` (audio) — lower acousticness correlates with higher popularity
3. `energy` (audio) — high-energy tracks trend more popular
4. `compound_sentiment` (sentiment) — *highest-ranked sentiment feature*
5. `valence_sentiment_gap` (engineered) — contributes meaningfully

Permutation importance confirms that `compound_sentiment` and `valence_sentiment_gap` both add genuine predictive value beyond audio features alone.

---

## 4. Key Findings

1. **Lyrics sentiment matters, but modestly.** Songs with positive lyrics are slightly more popular on average, but the effect size is small. Sentiment alone cannot predict a hit.

2. **Audio features dominate, but sentiment adds value.** The best regression model (Random Forest) improves R² from 0.17 (audio only) to ~0.19 when sentiment features are added—a consistent, if incremental, improvement.

3. **Emotional mismatch is a real signal.** Songs with a high valence–sentiment gap (e.g., happy-sounding music with sad lyrics) are slightly more popular, supporting H3. This "bittersweet" quality may be part of what makes certain songs resonate broadly.

4. **Popularity is fundamentally hard to predict from features alone.** Even the best model achieves R² ≈ 0.19, confirming that popularity depends heavily on external factors (artist recognition, playlist placement, release timing) not captured in audio/lyric features.

5. **Classification is more tractable than regression.** Identifying whether a song will be "popular" (binary) achieves ROC-AUC ~0.73—useful for filtering, if not precise prediction.

---

## 5. Limitations and Future Work

### Limitations

- **VADER is not music-specific.** VADER was trained on social media text. Music lyrics use metaphor, repetition, and unconventional grammar that can fool sentiment classifiers.
- **Language bias.** The primary dataset is English-heavy; sentiment analysis is less reliable for non-English lyrics.
- **Confounding variables.** Artist fame, playlist placement, release date, and social media virality are not in the data but are major drivers of popularity.
- **Popularity is a snapshot.** Spotify popularity scores change over time; the dataset captures a single moment.
- **Binary popularity threshold.** Defining "popular" as ≥ 60 is arbitrary; results may vary with different thresholds.

### Future Work

- **Deep learning on lyrics:** Use transformer-based models (e.g., fine-tuned BERT or a music-specific LLM) for richer sentiment representation.
- **Temporal analysis:** Track how a song's popularity evolves after release and model the trajectory rather than a single score.
- **Multi-modal models:** Combine audio waveforms, lyrics, and metadata (genre, release date, artist follower count) for a more comprehensive predictor.
- **Cross-language analysis:** Apply multilingual sentiment models to compare how emotional dynamics differ across language markets.
- **Causal inference:** Move beyond correlation—design a study that isolates the causal effect of lyrics sentiment on streaming behavior.

---

## 6. Repository Structure

```
Dsa210-spotify/
├── data/
│   ├── spotify_songs.csv        ← primary (lyrics + audio features, ~18K tracks)
│   └── dataset.csv              ← secondary (broader genre coverage, ~114K tracks)
├── notebook/
│   ├── EDA_and_Hypothesis_Tests.ipynb   ← Milestone 1 (April 14)
│   └── ML_Analysis.ipynb               ← Final ML methods (May 18)
├── figures/
│   ├── popularity_distribution.png
│   ├── sentiment_distribution.png
│   ├── valence_vs_sentiment.png
│   ├── popularity_by_sentiment_category.png
│   ├── correlation_heatmap.png
│   ├── regression_comparison.png        ← NEW
│   ├── roc_curves.png                   ← NEW
│   ├── confusion_matrices.png           ← NEW
│   ├── feature_importance.png           ← NEW
│   ├── permutation_importance.png       ← NEW
│   ├── valence_vs_sentiment_scatter.png ← NEW
│   └── gap_analysis.png                 ← NEW
├── requirements.txt
├── proposal_v2.docx
└── README.md
```

---

## 7. How to Reproduce

```bash
git clone https://github.com/adasarabil/Dsa210-spotify.git
cd Dsa210-spotify
pip install -r requirements.txt

# Run EDA and hypothesis tests
jupyter notebook notebook/EDA_and_Hypothesis_Tests.ipynb

# Run ML analysis
jupyter notebook notebook/ML_Analysis.ipynb
```

---

## 8. AI Usage Disclosure

This project used **Claude (Anthropic)** as an AI assistant throughout, in accordance with DSA 210 academic integrity guidelines. Specific uses include:

- **Dataset selection:** Claude helped evaluate Kaggle dataset options and identify the enrichment strategy (using VADER sentiment as an independent variable to complement Spotify's audio-derived valence).
- **Research question refinement:** Claude suggested framing the valence–sentiment gap as a third hypothesis.
- **Code scaffolding:** Claude provided starter code for VADER integration, Steiger's test implementation, and the ML pipeline structure.
- **Report structure:** Claude helped outline the final report sections.

All analysis decisions, model interpretation, and conclusions were made and validated by the student. No AI-generated text appears verbatim in the analysis sections.

---

*DSA 210 Spring 2026 — Sabancı University*
