from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from gensim.matutils import jensen_shannon
from scipy import spatial as scs
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist, squareform

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

    # Calculate perplexity score
    perplexity_score = lda_model.log_perplexity(doc_term_matrix)

    return lda_model, dictionary, coherence_score, perplexity_score


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

def autotune_lda_model(documents, min_topics=2, max_topics=10, step=1):
    coherence_scores = []
    models = []

    for num_topics in range(min_topics, max_topics + 1, step):
        lda_model, _, coherence_score, _ = train_lda_model(documents, num_topics)
        coherence_scores.append(coherence_score)
        models.append(lda_model)

    # Creating coherence scores plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(min_topics, max_topics + 1, step), coherence_scores)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Scores for Different Numbers of Topics')

    # Select the model with the highest coherence score
    best_model_index = coherence_scores.index(max(coherence_scores))
    best_model = models[best_model_index]
    best_num_topics = range(min_topics, max_topics + 1, step)[best_model_index]

    # Get the dictionary used for the best model
    _, dictionary, _, _ = train_lda_model(documents, best_num_topics)

    return best_model, best_num_topics, dictionary, plt

def convergence_lda_model(documents, start_topics, end_topics, step, start_words, end_words, words_step):
    results = []
    coherence_scores = []
    perplexity_scores = []

    # Iterate over the range of topics and words
    for num_topics in range(start_topics, end_topics + 1, step):
        for num_words in range(start_words, end_words + 1, words_step):
            lda_model, dictionary, coherence_score, perplexity_score = train_lda_model(documents, num_topics)
            topics = get_lda_topics(lda_model, dictionary, num_words)
            results.append({
                'num_topics': num_topics,
                'num_words': num_words,
                'coherence': coherence_score,
                'perplexity': perplexity_score,
                'topics': topics
            })
            coherence_scores.append(coherence_score)
            perplexity_scores.append(perplexity_score)

    # Creating coherence and perplexity scores plot
    plt.figure(figsize=(12, 6))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Coherence plot
    plt.subplot(1, 2, 1)
    plt.plot(range(len(coherence_scores)), coherence_scores, marker='o', color='b')
    plt.xlabel('Configuration Index')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Scores for Different Configurations')

    # Perplexity plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(perplexity_scores)), perplexity_scores, marker='o', color='r')
    plt.xlabel('Configuration Index')
    plt.ylabel('Perplexity Score')
    plt.title('Perplexity Scores for Different Configurations')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot()

    return results

