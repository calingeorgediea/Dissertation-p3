from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
from gensim.models import CoherenceModel
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

def train_lda_model(documents, num_topics=5):
    # Preprocess and tokenize the documents
    tokenized_documents = [preprocess_text(doc) for doc in documents]
    
    # Create a dictionary from the tokenized documents
    dictionary = corpora.Dictionary(tokenized_documents)

    # Create a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    # Train the LDA model
    lda_model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)

    # Calculate coherence score
    coherence_score = calculate_coherence_score(lda_model, tokenized_documents, dictionary)

    return lda_model, dictionary, coherence_score

def calculate_coherence_score(lda_model, tokenized_documents, dictionary):
    """
    Calculate the coherence score for the LDA model.

    Args:
    lda_model: The trained LDA model.
    tokenized_documents: List of tokenized documents.
    dictionary: Gensim dictionary object.

    Returns:
    Coherence score.
    """
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

def get_lda_topics(lda_model, dictionary, num_words=5):
    return [lda_model.show_topic(topicid, num_words) for topicid in range(lda_model.num_topics)]

