
# Resume Screening Project

This project demonstrates two approaches to automate resume screening using NLP and machine learning. Both notebooks classify resumes as either:

- "Fit – Move forward with interview"
- "Not a good fit"

The dataset used is from Kaggle: [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

---

## Files Included

- `model1_RB.ipynb` – Uses a simple keyword threshold to label candidates.
- `model2_Z-Score.ipynb` – Uses a statistical Z-score approach for more adaptive labeling.

---

## How It Works

### 1. Data Cleaning
- Strips HTML tags, punctuation, numbers
- Converts text to lowercase
- Removes stopwords using NLTK

### 2. Keyword Scoring
- Uses a list of relevant data science keywords from `keywords.csv`
- For each resume, counts how many keywords are present (`KeywordScore`)

---

## Method 1: Rule-Based Threshold

- Any resume with 3 or more keyword matches is labeled as `"Fit"`.
- Otherwise labeled as `"Not a good fit"`.
- Simple and easy to tune.

### Model
- Text vectorized using `CountVectorizer`
- Logistic Regression is trained to classify based on cleaned text

---

## Method 2: Z-Score Based Labeling

- Computes the **Z-score** of each resume’s `KeywordScore`
- A resume is labeled `"Fit"` if its z-score is ≥ 0.5
- This method adapts based on the distribution of scores across the dataset
- Histogram of Z-scores is plotted to visualize threshold impact

---

## Requirements

Make sure to install required libraries:
```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

Also, download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

---

## Output

Each notebook:
- Trains and evaluates a logistic regression model
- Prints classification metrics (precision, recall, f1-score)
- Displays each resume’s final decision
- Optionally saves results to CSV

---

## Note

Make sure the following files are in the same directory:
- `resume_dataset.csv` – The resume text data
- `keywords.csv` – Your custom keyword list for scoring

---

## Author
Developed as part of a data science firm simulation for academic purposes by Sanjana Suresh.

