from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Check if text is a non-empty string
    if not isinstance(text, str) or not text.strip():
        return []

    # Tokenize the document
    tokens = word_tokenize(text.lower())

    # Remove stopwords and words with length <= 3
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 3]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def train_nmf_model(documents, num_topics=5):
    # Preprocess and tokenize the documents
    tokenized_documents = [' '.join(preprocess_text(doc)) for doc in documents]

    # Create a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(tokenized_documents)

    # Train the NMF model
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_model.fit(tfidf_matrix)

    return nmf_model, vectorizer.get_feature_names_out()

def get_nmf_topics(nmf_model, feature_names, num_words=5):
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append((topic_idx, " ".join(top_words)))
    
    return topics

def compute_coherence_score(documents, nmf_model, feature_names, num_words=5):
    """
    Compute the coherence score for the NMF model.

    :param documents: List of original (preprocessed) documents
    :param nmf_model: Trained NMF model
    :param feature_names: List of feature names (words) from TF-IDF
    :param num_words: Number of top words to consider for each topic
    :return: Coherence score
    """
    # Get top words for each topic
    top_words_per_topic = []
    for topic in nmf_model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        top_words_per_topic.append(top_words)

    # Preprocess documents for coherence calculation
    preprocessed_documents = [simple_preprocess(doc) for doc in documents]

    # Create a dictionary and corpus for coherence calculation
    dictionary = corpora.Dictionary(preprocessed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]

    # Create CoherenceModel and compute the coherence score
    coherence_model = CoherenceModel(topics=top_words_per_topic, texts=preprocessed_documents, dictionary=dictionary, coherence='c_v')
    
    return coherence_model.get_coherence()