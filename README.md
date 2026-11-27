# Fake News Detection

An NLP and Machine Learning project that analyzes news headlines and full-length articles and predicts whether they are **LIKELY REAL** or **LIKELY FAKE**.

## 🚀 Live Demo

**Try the deployed application:**

🔗 [Fake News Detection – Live App](https://fake-news-detection-advanced.streamlit.app/)

Explore the application:

- **News Analyzer:** Analyze news headlines and full-length articles.
- **Model Performance:** View model evaluation metrics.
- **Project Information:** Explore project details and methodology.

**Note:** Predictions are based on patterns learned from the ISOT dataset. They are ML-based classifications, not independent verification of news authenticity.

## Project Overview

Fake News Detection is a Python-based machine learning application built using TF-IDF feature extraction and multiple classical ML algorithms.

The project supports two input types:
- News headlines
- Full news articles

A mixed-input training approach is used so that the model can learn from both headlines and full articles instead of being trained only on long-form text.

The application provides predictions through an interactive Streamlit interface.

## Features

- News headline and full-article classification
- Text preprocessing and validation
- Word-level and character-level TF-IDF features
- Four ML model implementations and comparison
- Separate training, validation, and test datasets
- Mixed-input Linear SVM model
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Streamlit dashboard for news analysis
- Automated software tests using Pytest
- Saved model artifacts and evaluation reports

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP / Feature Extraction | TF-IDF |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Model Storage | Joblib |
| Testing | Pytest |

## Dataset

This project uses the **ISOT Fake News Dataset**.

Original dataset files:
- `Fake.csv`
- `True.csv`

Dataset processing includes:
- Empty-record removal
- Duplicate removal
- Label consistency checks
- Text preprocessing
- Train, validation, and test splitting
- Duplicate fingerprint checks to reduce data leakage across splits

Processed dataset statistics:

| Dataset Split | Records |
|---|---:|
| Training | 27,178 |
| Validation | 3,883 |
| Testing | 7,766 |
| **Total** | **38,827** |

Label mapping:
- `0` – REAL
- `1` – FAKE

**Dataset note:** The ISOT dataset is intended for research and educational use. Refer to the original dataset terms and attribution requirements before redistribution or commercial use. Raw dataset files are not included in this repository.

## Machine Learning Models

The following four classical ML algorithms are trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)
4. Random Forest Classifier

### Feature Engineering

The project uses two TF-IDF representations:

- Word-level TF-IDF: unigrams and bigrams
- Character-level TF-IDF: character n-grams

Both feature representations are combined to capture word patterns and character-level text patterns.

### Mixed-Input Training

The original full-text model showed that headline-only predictions needed separate attention.

To address this, a mixed-input model was developed using both:
- Cleaned headlines
- Combined full article text

A separate Linear SVM model and TF-IDF vectorizer are trained for this approach.

## Model Evaluation

### Mixed-Input Linear SVM – Test Results

| Input Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Headline Only | 96.42% | 96.72% | 95.47% | 96.09% | 99.43% |
| Full Article | 99.20% | 99.30% | 98.97% | 99.13% | 99.96% |

### Original Full-Text Model Comparison

| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 99.34% | 99.55% | 99.02% | 99.29% |
| Naive Bayes | 95.75% | 96.86% | 93.83% | 95.32% |
| Linear SVM | 99.81% | 99.89% | 99.69% | 99.79% |
| Random Forest | 99.48% | 99.66% | 99.22% | 99.44% |

**Important:** These results are measured on held-out portions of the ISOT dataset. They do not guarantee the same performance on new, real-world news. Dataset-specific patterns and source bias may influence the results.

## Project Structure

```text
Fake_News_Detection_Advanced/
│
├── app.py
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── tests/
├── requirements.txt
├── requirements-dl.txt
├── pytest.ini
├── pyproject.toml
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/24A31A05EG/FAKE_NEWS_DETECTION.git
cd FAKE_NEWS_DETECTION
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the ISOT dataset

Place the original `Fake.csv` and `True.csv` files inside:

```text
data/raw/
```

### 5. Run data preprocessing

```bash
python -m src.data.pipeline --input data/raw --name isot
```

### 6. Train the models

```bash
python -m src.models.train_models
python -m src.models.train_mixed_input
```

### 7. Evaluate the models

Use the evaluation scripts and reports included in the project to inspect model performance.

### 8. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## Automated Testing

The project includes automated tests implemented using Pytest.

Run the complete test suite:

```bash
python -m pytest -v
```

The test suite covers:

- Configuration loading
- Input validation
- Dataset loading and preprocessing
- Duplicate detection and data splitting
- Processed dataset artifacts
- Trained model artifacts
- Model predictions
- Prediction edge cases

**Latest test verification:**
- Original project environment: 50 passed
- Fresh GitHub clone: 48 passed, 2 skipped

The two skipped tests require the ISOT processed dataset files, which are intentionally not included in the GitHub repository.

## Limitations

- Predictions are based on patterns learned from the ISOT dataset.
- The application does not independently verify news against live news sources.
- Model performance on unseen real-world news may differ from the reported test results.
- Dataset-specific patterns and source bias may affect classification.
- The application provides ML-based predictions, not guaranteed factual verification.

## Future Enhancements

- Deep Learning model integration
- Live news source and evidence retrieval
- Additional dataset evaluation
- Improved multilingual news analysis
- Explainable AI techniques
- Advanced prediction analysis

## Author

**Majji Kavya**

B.Tech – Computer Science and Engineering

Pragati Engineering College

GitHub: [24A31A05EG](https://github.com/24A31A05EG)

---

## Project Links

- **GitHub Repository:** [FAKE_NEWS_DETECTION](https://github.com/24A31A05EG/FAKE_NEWS_DETECTION)
- **Live Application:** [Fake News Detection – Streamlit](https://fake-news-detection-advanced.streamlit.app/)
