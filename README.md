# Tweet Hate Speech Classifier

Fine-tuning **BERTweet** on ~25k tweets to classify hate speech, offensive language, and neutral content — achieving a weighted F1-score of **0.9132**.

---

## Motivation

Social media moderation at scale is a hard problem. I wanted to see how well a transformer model pre-trained specifically on Twitter language could handle the noisy, slang-heavy nature of tweets. 

---

## Approach

- **Preprocessing** — Custom tweet normalization: usernames → `@USER`, URLs → `HTTPURL`, emojis → text tokens via `demojize`
- **Model** — [`vinai/bertweet-base`](https://huggingface.co/vinai/bertweet-base) fine-tuned for 3-class sequence classification
- **Training** — AdamW (lr=2e-5), linear warmup scheduler, 3 epochs, batch size 16, 90/10 train/val split
- **Why BERTweet?** — Pre-trained on 850M English tweets, it handles Twitter slang and abbreviations far better than standard BERT

---

## Result

| Metric | Score |
|---|---|
| Weighted F1 | **0.9132** |

---

## Structure

```
tweet-hate-speech-classifier/
├── tweet_classifier_bertweet.ipynb   # Training & inference pipeline
├── train.csv                         # Training data (not included — not publicly redistributable)
└── test_no_label.csv                 # Test data (not included)
```

---

## Getting Started

Open the notebook in Google Colab (GPU runtime recommended), upload your `train.csv` and run all cells. Training takes ~15–20 min on a T4 GPU. Predictions are exported to `test_with_label.csv`.

```
pip install torch transformers scikit-learn pandas numpy emoji==0.6.0 nltk
```

---

## Author

**Julian Steinhauser**
