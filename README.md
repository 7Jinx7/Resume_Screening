
# Resume Screening Project

**The goal of this project was to use natural language processing and data analytics (like Z-score analysis) to automatically screen resumes and identify top candidates from a large dataset.**

This project demonstrates two approaches to automate resume screening using NLP and machine learning. Both notebooks classify resumes as either:

- "Fit – Move forward with interview"
- "Not a good fit"

The dataset used is from Kaggle: [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

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

## Files Included

- `model1_RB.ipynb` – Uses a simple keyword threshold to label candidates.
- `model2_Z-Score.ipynb` – Uses a statistical Z-score approach for more adaptive labeling.
- Please also download the `cleaned_dataset.csv` and the `keywords.csv` before running the ipynb notebooks.

---

## Part 1 of project: Setup

### 1. Data Cleaning
- Original dataset is from Kaggle
- Strip HTML tags, punctuation, numbers
- Convert alll text to lowercase
- Remove stopwords using NLTK
- COMPLETED: THIS IS ALREADY DONE AND SAVED IN `cleaned_dataset.csv`

### 2. Keyword Scoring
- Uses a list of relevant data science keywords from `keywords.csv`
- For each resume, counts how many keywords are present (`KeywordScore`)
- COMPLETED: THIS IS ALREADY DONE AND SAVED IN `keywords.csv`

---

## What to run

- If you are looking to hire more people (e.g: needs of a small start-up), we recommend you try the model1_RB.ipynb notebook.
- If you ate looking to hire few people (e.g: interns for a big company), we recommend you try the model2_Z-Score.ipynb notebook.
- **If you are an instructor or a grader, we recommend you run both!**
- More information on the models below. 
---

## Part 2 of project: Model building and training

## Method 1: Rule-Based Threshold

- Any resume with 3 or more keyword matches is labeled as `"Fit"`.
- Otherwise labeled as `"Not a good fit"`.
- Simple and easy to tune.

### Model
- Text vectorized using `CountVectorizer`
- Logistic Regression is trained to classify based on cleaned text
- We picked a threshold of 3 keywords in the rule-based method as a simple heuristic to identify candidates with basic relevant skills, though this number can be easily customized based on job requirements.

---

## Method 2: Z-Score Based Labeling

- Computes the **Z-score** of each resume’s `KeywordScore`
- A resume is labeled `"Fit"` if its z-score is ≥ 0.5
- This method adapts based on the distribution of scores across the dataset
- Histogram of Z-scores is plotted to visualize threshold impact

- The Z-score was computed by taking each resume's keyword match count, subtracting the average keyword count across all resumes, and dividing by the standard deviation.

  - Mathematically: Z = (KeywordScore - MeanKeywordScore) / StandardDeviation
  - This tells us how far a resume’s keyword count is from the average, in units of standard deviation, helping us identify resumes that         stand out.

- We chose a Z-score threshold of 0.5 to select resumes that have keyword scores at least half a standard deviation above the mean, indicating above-average relevance without being overly restrictive.

---

## Output

Each notebook:
- Trains and evaluates a logistic regression model
- Prints classification metrics (precision, recall, f1-score)
- Displays each resume’s final decision
- Saves results to CSV

---

## Note

Make sure the following files are in the same directory:
- `resume_dataset.csv` – The resume text data
- `keywords.csv` – Your custom keyword list for scoring

---

## Author
Developed as part of a data science classd for academic purposes by Sanjana Suresh (primary) and Nimisha Pant.

