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

def find_best_lda_model(documents, max_topics=10, max_words_per_topic=10):
    # Preprocess and tokenize the documents
    tokenized_documents = [preprocess_text(doc) for doc in documents]
    
    # Create a dictionary from the tokenized documents
    dictionary = corpora.Dictionary(tokenized_documents)

    # Create a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    best_lda_model = None
    best_coherence_score = -1.0  # Initialize with a low value
    best_num_topics = 0
    best_num_words_per_topic = 0

    for num_topics in range(1, max_topics + 1):
        for num_words_per_topic in range(1, max_words_per_topic + 1):
            lda_model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)
            coherence_score = calculate_coherence_score(lda_model, tokenized_documents, dictionary)

            if coherence_score > best_coherence_score:
                best_lda_model = lda_model
                best_coherence_score = coherence_score
                best_num_topics = num_topics
                best_num_words_per_topic = num_words_per_topic

    # Plot convergence data for the best model
    plot_convergence(best_lda_model, tokenized_documents, dictionary)

    return best_lda_model, dictionary, best_num_topics, best_num_words_per_topic, best_coherence_score

def plot_coherence_scores(coherence_scores, num_topics_range, num_words_per_topic_range):
    # Create a 2D grid of coherence scores
    scores_grid = [[coherence_scores[i * len(num_words_per_topic_range) + j] for j in range(len(num_words_per_topic_range))] for i in range(len(num_topics_range))]

    # Plot the heatmap of coherence scores
    plt.figure(figsize=(10, 6))
    plt.imshow(scores_grid, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(num_words_per_topic_range)), num_words_per_topic_range)
    plt.yticks(range(len(num_topics_range)), num_topics_range)
    plt.xlabel('Words per Topic')
    plt.ylabel('Number of Topics')
    plt.title('Coherence Scores Heatmap')
    plt.show()
    
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

