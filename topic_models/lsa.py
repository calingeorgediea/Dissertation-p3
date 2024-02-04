from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_20newsgroups 
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim

def load_data():
    
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return newsgroups


def preprocess_text(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token.isalpha():
            result.append(lemmatize_stemming(token))
    return result

def lemmatize_stemming(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text, pos='v') # 'v' stands for verb, can change based on context


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

    reconstruction_error = calculate_reconstruction_error(X, lsa, num_topics)

    return topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity, lsa, vectorizer, reconstruction_error

def compute_lsa_coherence_score(text_data, lsa_model, vectorizer, num_topics, num_words=5):
    # Preprocess the text data
    preprocessed_text_data = [preprocess_text(doc) for doc in text_data]

    # Create a dictionary from the preprocessed text data
    dictionary = corpora.Dictionary(preprocessed_text_data)

    # Create a corpus using the dictionary
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_text_data]

    # Extract top words for each topic
    top_words_per_topic = []
    for topic_idx in range(num_topics):
        topic_terms = lsa_model.components_[topic_idx]
        top_terms_idx = topic_terms.argsort()[-num_words:][::-1]
        top_words = [vectorizer.get_feature_names_out()[i] for i in top_terms_idx]
        top_words_per_topic.append(top_words)

    # Compute coherence score
    coherence_model = CoherenceModel(topics=top_words_per_topic, texts=preprocessed_text_data, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    return coherence_score

def calculate_reconstruction_error(X, lsa_model, num_topics):
    """
    Calculate the reconstruction error for the LSA model.
    
    Args:
    X: The original document-term matrix.
    lsa_model: Trained LSA model.
    num_topics: Number of topics.

    Returns:
    Reconstruction error.
    """
    # Reconstruct the matrix using LSA components
    reconstructed_X = lsa_model.transform(X) @ lsa_model.components_

    # Calculate the Frobenius norm (difference between the original and reconstructed matrix)
    reconstruction_error = np.linalg.norm(X - reconstructed_X, 'fro')

    return reconstruction_error
