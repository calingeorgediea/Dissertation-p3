from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np

def perform_lsa(text_data, num_topics, num_words, matrix_type='raw'):

    if matrix_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = CountVectorizer(stop_words='english')
    
    X = vectorizer.fit_transform(text_data)

    # Document-term matrix
    doc_term_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

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

    return topics, doc_term_matrix, explained_variance, topic_term_matrix, sparsity