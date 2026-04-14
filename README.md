DSA 210 – What Makes a Song Popular on Spotify?
Student: Ada Derviş Sarabil – 34362
Course: DSA 210 Introduction to Data Science, Spring 2026

Research Question
Does the emotional content of a song's lyrics — measured through NLP sentiment analysis — explain Spotify popularity beyond what audio features alone can capture? And when a song's musical tone and its lyrics point in opposite directions emotionally (e.g. upbeat melody but dark lyrics), does that tension itself predict higher popularity?

Hypotheses
H1: Songs with positive lyrics sentiment have significantly higher Spotify popularity than songs with negative lyrics sentiment.

Formally: The median popularity score of positive-sentiment songs is significantly higher than that of negative-sentiment songs.
Test: Mann-Whitney U test (α = 0.05)

H2: Spotify's audio-derived valence is a weaker predictor of popularity than VADER lyrics sentiment score.

Formally: The Spearman correlation between lyrics sentiment and popularity is significantly stronger than the correlation between valence and popularity.
Test: Steiger's test for dependent correlations (α = 0.05)

H3: Songs with a high valence–sentiment gap (emotionally mismatched audio and lyrics) have higher popularity than low-gap songs.

Formally: The median popularity of high-gap songs is significantly higher than low-gap songs.
Test: Mann-Whitney U test (α = 0.05)
Gap metric: valence_sentiment_gap = |valence − normalized_sentiment|


Datasets
DatasetSourceDescriptionSpotify Tracks DatasetKaggle – maharshipandya~114,000 tracks with audio features (danceability, energy, valence, tempo, etc.) and popularity scores (0–100)Audio Features and Lyrics of Spotify SongsKaggle – imuhammadSpotify tracks with full song lyrics, joined on track_id
How the enrichment dataset is used
The lyrics dataset is joined to the primary dataset via track_id. From the lyrics column, a VADER sentiment analysis (via nltk.sentiment.vader) is run on each track to produce a compound_sentiment score ranging from −1 (most negative) to +1 (most positive). This score is entirely independent of Spotify's own valence feature, which is derived from audio signals rather than text — making it a genuine external enrichment variable.
Two new features are derived from this join:

compound_sentiment — lyrics-based emotional polarity score
valence_sentiment_gap — absolute difference between valence and normalized sentiment, capturing emotional mismatch between sound and lyrics

Instrumental tracks (missing lyrics) are flagged separately and analyzed as a subgroup rather than excluded entirely.

Repository Structure
your-repo/
├── data/
│   ├── spotify_tracks.csv
│   └── spotify_lyrics.csv
├── notebooks/
│   └── EDA_and_Hypothesis_Tests.ipynb
├── figures/
│   ├── sentiment_distribution.png
│   ├── valence_vs_sentiment.png
│   ├── popularity_by_sentiment_category.png
│   ├── valence_sentiment_gap.png
│   └── correlation_heatmap.png
├── requirements.txt
└── README.md

How to Run
bashpip install -r requirements.txt
mkdir -p figures
jupyter notebook notebooks/EDA_and_Hypothesis_Tests.ipynb
Then run all cells (Cell → Run All).

Requirements
See requirements.txt. Main dependencies: pandas, numpy, matplotlib, seaborn, scipy, nltk, jupyter.
