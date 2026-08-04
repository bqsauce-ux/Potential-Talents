# Potential Talents: Candidate Ranking Using NLP and RankNet

## Project Overview

This project develops an intelligent candidate ranking system for Human Resources recruitment. Instead of relying solely on keyword matching, the notebook combines multiple Natural Language Processing (NLP) techniques with a learning-to-rank neural network to identify candidates whose job titles best match HR-related positions.

---

## Objective

The goal of this project is to:

- Rank candidates based on semantic similarity to Human Resources roles.
- Compare multiple NLP approaches for measuring candidate relevance.
- Generate a weighted fitness score using several similarity metrics.
- Train a RankNet model to improve candidate ordering.
- Produce a final ranked list of candidates.

---

## Dataset

The dataset contains candidate information including:

- Candidate ID
- Job Title
- Current Location
- Number of Connections
- Existing HR Fit Score

---

## Exploratory Data Analysis (EDA)

The notebook performs several exploratory analyses:

- Dataset inspection
- Missing value detection
- Descriptive statistics
- Candidate location analysis
- Distribution of LinkedIn connections
- Word clouds of locations
- Word clouds of connection counts

These visualizations help understand the overall candidate pool before feature engineering.

---

## Feature Engineering

Several similarity features are generated for every candidate.

### 1. BERT Semantic Similarity

- Uses Sentence Transformers (`all-MiniLM-L6-v2`)
- Encodes candidate job titles
- Encodes HR target phrases
- Computes cosine similarity
- Stores the highest similarity as **bert_score**

---

### 2. Keyword Matching

A simple keyword-based approach is implemented to determine whether candidate job titles contain Human Resources related terms.

---

### 3. FastText Embeddings

The notebook trains a FastText model on candidate job titles.

For each candidate:

- Words are converted into embeddings
- Word embeddings are averaged
- Cosine similarity is calculated against the HR target
- Results are stored as **fasttext_score**

---

### 4. TF-IDF Similarity

TF-IDF vectors are generated for all job titles.

The notebook:

- Vectorizes text
- Calculates cosine similarity
- Generates a **tfidf_score** for every candidate

---

## Combined Fitness Score

Instead of relying on one NLP technique, multiple similarity scores are combined into a weighted score.

The weighted score includes:

- BERT similarity
- TF-IDF similarity
- FastText similarity
- Existing HR fitness score

This provides a more robust estimate of candidate relevance.

---

## Learning to Rank (RankNet)

After feature engineering, the notebook trains a RankNet neural network.

The process includes:

- Creating pairwise candidate comparisons
- Building a three-layer neural network
- Training with Binary Cross Entropy Loss
- Optimizing with Adam
- Predicting ranking scores
- Sorting candidates from best to worst

---

## Technologies Used

- Python
- Pandas
- NumPy
- Sentence Transformers
- Gensim FastText
- Scikit-learn
- PyTorch
- Matplotlib
- WordCloud

---

## Project Workflow

1. Load candidate dataset
2. Perform exploratory data analysis
3. Generate semantic similarity features
4. Compute TF-IDF similarity
5. Train FastText embeddings
6. Calculate weighted fitness scores
7. Generate pairwise ranking data
8. Train RankNet
9. Rank candidates
10. Produce final recommendations

---

## Results

The notebook produces:

- BERT similarity scores
- FastText similarity scores
- TF-IDF similarity scores
- Weighted fitness scores
- RankNet predictions
- Final candidate rankings

---

## Future Improvements

Potential enhancements include:

- Fine-tuning BERT models
- Incorporating resume text instead of job titles alone
- Adding years of experience
- Including education and skills
- Experimenting with LambdaRank or LambdaMART
- Deploying the ranking model as a web application

---

## Repository Structure

```
Potential-Talents/
│
├── Potential Talents.ipynb
├── README.md
├── requirements.txt
└── data/
```

---

## Author

Machine Learning Candidate Ranking Project using NLP and Learning-to-Rank techniques.
