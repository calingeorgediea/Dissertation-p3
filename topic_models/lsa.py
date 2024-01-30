from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_20newsgroups 

def load_data():
    
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups

def perform_lsa(text_data, num_topics, num_words, matrix_type='raw'):
    # Check if text_data is a list of strings
    if not all(isinstance(doc, str) for doc in text_data):
        # If not, assume it's a list of lists and join the tokens into strings
        text_data = [' '.join(doc) for doc in text_data]

    # Choose the vectorizer based on the specified matrix type
    if matrix_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = CountVectorizer(stop_words='english')

    # Create the document-term matrix
    X = vectorizer.fit_transform(text_data)
    doc_term_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Perform LSA
    lsa = TruncatedSVD(n_components=num_topics)
    lsa.fit(X)
 
    # Explained variance
    explained_variance = lsa.explained_variance_ratio_

    # Topic-term matrix
    topic_term_matrix = pd.DataFrame(lsa.components_, index=[f"Topic {i+1}" for i in range(num_topics)], columns=vectorizer.get_feature_names_out())

    # Calculating sparsity
    sparsity = (1.0 - np.count_nonzero(doc_term_matrix) / doc_term_matrix.size) * 100

    terms = vectorizer.get_feature_names_out()
    
    topics = []
    for i, comp in enumerate(lsa.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:num_words]
        topics.append("Topic " + str(i+1) + ": " + ", ".join([t[0] for t in sorted_terms]))

    lsa = TruncatedSVD(n_components=num_topics)
    lsa.fit(X)

    # U matrix (documents in the latent topic space)
    U_matrix = lsa.transform(X)

    # Sigma matrix (diagonal matrix of singular values)
    Sigma_matrix = np.diag(lsa.singular_values_)

    # V^T matrix (topics in the latent feature space)
    VT_matrix = lsa.components_

    return topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity