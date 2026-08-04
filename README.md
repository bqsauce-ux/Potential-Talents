# Potential-Talents

Overview

This project builds a candidate ranking pipeline for identifying talentthat best matches Human Resources roles. Multiple NLP similaritytechniques are combined into a weighted fitness score, which is thenused to train a RankNet learning-to-rank model.

Workflow

Exploratory Data Analysis (EDA)

Examine dataset statistics.

Analyze connection counts.

Visualize locations and connections using word clouds.

Semantic Matching (BERT)

Uses the all-MiniLM-L6-v2 SentenceTransformer.

Compares job titles against:

Aspiring Human Resources

Seeking Human Resources

Stores the highest cosine similarity as bert_score.

Keyword Matching

Traditional keyword-based comparison of job titles.

FastText Similarity

Trains a FastText model on job titles.

Generates sentence embeddings by averaging word vectors.

Computes cosine similarity with the HR target phrase.

Saves results as fasttext_score.

TF-IDF Similarity

Builds TF-IDF vectors for job titles.

Measures cosine similarity to the HR target phrase.

Stores results as tfidf_score.

Weighted Fitness Score

Combines BERT, TF-IDF, FastText, and HR fitness metrics into asingle overall candidate fitness score.

Learning to Rank

Creates pairwise candidate comparisons.

Implements a three-layer RankNet neural network in PyTorch.

Trains using binary cross-entropy loss and the Adam optimizer.

Produces a final ranking score for each candidate.

Technologies

Python

pandas

NumPy

Sentence Transformers

FastText (Gensim)

scikit-learn

PyTorch

Output

The notebook generates similarity scores from multiple NLP methods,combines them into a weighted fitness score, trains a RankNet model, andoutputs candidates ranked from most to least suitable for Human
