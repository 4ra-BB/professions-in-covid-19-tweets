# Detecting Profession Mentions in COVID-19 Tweets

Binary classification of Spanish tweets to identify mentions of professions during the COVID-19 pandemic, using fine-tuned TwHIN-BERT (Transformers).

## The problem

During the COVID-19 pandemic, certain professions faced disproportionate health risks. Detecting profession mentions in social media enables analysis of which occupations are being discussed in the context of the crisis, complementing traditional epidemiological surveillance.

## Dataset

Data from the [ProfNER shared task](https://temu.bsc.es/smm4h-spanish), hosted on [Hugging Face](https://huggingface.co/datasets/luisgasco/profner_classification_master). Each tweet is labeled as:

- `1` — mentions at least one profession
- `0` — does not mention a profession

## Approach

| Step | Method |
|---|---|
| Model | [TwHIN-BERT](https://huggingface.co/Twitter/twhin-bert-base) — multilingual BERT pre-trained on tweets |
| Task | Binary sequence classification |
| Training | Fine-tuning with Hugging Face Trainer (4 epochs, lr=2e-5) |
| Evaluation | F1 score, accuracy, classification report |

### Why TwHIN-BERT?

1. **Domain match**: pre-trained on tweets, so it already understands informal text, hashtags, and mentions
2. **Language**: multilingual, includes Spanish
3. **Efficiency**: BERT-base architecture, fine-tunable on ~6K examples without heavy compute

## Repository structure

```
├── README.md
├── notebooks/
│   └── profner_tweet_classification.ipynb
└── .gitignore
```

The dataset is loaded directly from Hugging Face inside the notebook. No local data files needed.

## How to run

The easiest way is Google Colab (GPU recommended for training):

1. Upload `profner_tweet_classification.ipynb` to Colab
2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells

## Tools

Python · Hugging Face Transformers · PyTorch · scikit-learn · matplotlib · seaborn · NLTK

## Author

**Laura Benkel Brander** — Sociologist & Data Scientist  
[LinkedIn](https://www.linkedin.com/in/laurabenkel)
