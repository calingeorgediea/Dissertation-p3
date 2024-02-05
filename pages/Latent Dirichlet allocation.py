import streamlit as st
from topic_models import lsa  # Importing LSA from the topic_models package
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from topic_models import lda
from topic_models import nmf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import models.history as history
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit.components.v1 as components
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gpt2 import display_gpt2_page
import matplotlib.pyplot as plt
import utils.preprocess as utils
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


user_input_text = st.text_area("Enter your text for analysis (Optional)")
num_topics = st.sidebar.slider("Number of Topics", 1, 20, 5)


start_topics = st.sidebar.slider("Start Number of Topics", 1, 20, 1)
end_topics = st.sidebar.slider("End Number of Topics", 1, 20, 3)
topics_step = st.sidebar.slider("Step for Number of Topics", 1, 5, 1)

start_words = st.sidebar.slider("Start Number of Words per Topic", 1, 20, 1)
end_words = st.sidebar.slider("End Number of Words per Topic", 1, 20, 3)
words_step = st.sidebar.slider("Step for Number of Words per Topic", 1, 5, 1)

@st.cache_data
def load_cached_data(subset_size=20):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    if subset_size is not None and subset_size < len(data):
        return random.sample(data, subset_size)

    processed_data = [utils.preprocess_text(doc) for doc in data]
    st.write(processed_data)
    return processed_data

def load_new_data(subset_size=20):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    if subset_size is not None and subset_size < len(data):
        return random.sample(data, subset_size)

    processed_data = [utils.preprocess_text(doc) for doc in data]
    st.write(processed_data)
    return processed_data

def preprocess_documents(documents):
    stop_words = set(stopwords.words('english'))
    preprocessed_docs = []

    for doc in documents:
        # Tokenize and remove stop words
        tokens = word_tokenize(doc)
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        preprocessed_docs.append(filtered_tokens)

    return preprocessed_docs

def display_lda_results(topics):
    st.write("LDA Topics:")
    for i, topic in enumerate(topics):
        topic_words = ", ".join([word for word, _ in topic])
        st.write(f"Topic {i + 1}: {topic_words}")

def display_lda_topic_charts(topics):
    for i, topic in enumerate(topics):
        words, probabilities = zip(*topic)
        topic_df = pd.DataFrame({'Word': words, 'Probability': probabilities})
        st.write(f"Topic {i + 1}")
        st.bar_chart(topic_df.set_index('Word'))
        
def fdisplay_lda_visualization(lda_model, corpus, dictionary):
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    components.html(html_string, width=800, height=600, scrolling=True)



def display_results(topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity):
    st.write("LSA Topics:")
    for topic in topics:
        st.write(topic)

    st.write("Analysis")
    st.write("Document-Term Matrix:")
    st.dataframe(doc_term_matrix)
    st.write("Explained Variance by Each Topic:")
    st.bar_chart(explained_variance)
    st.write("Topic-Term Matrix:")
    st.dataframe(topic_term_matrix)
    st.write(f"Matrix Sparsity: {sparsity:.2f}%")
    


    # Display SVD Matrices
    if U_matrix is not None:
        st.write("U Matrix (Document-Topic Matrix):")
        st.dataframe(pd.DataFrame(U_matrix))
    if Sigma_matrix is not None:
        st.write("Sigma Matrix (Diagonal Matrix of Singular Values):")
        st.dataframe(pd.DataFrame(Sigma_matrix))
    if VT_matrix is not None:
        st.write("V^T Matrix (Topic-Term Matrix):")
        st.dataframe(pd.DataFrame(VT_matrix))

def main():
    st.title("Topic Modeling Dashboard")

    # Initialize session state for storing results
    if 'results' not in st.session_state:
        st.session_state['results'] = {}

        # In your sidebar settings in the main function
    use_cache = st.sidebar.checkbox("Cache Documents", value=False)

    dataset_size = st.sidebar.slider("Select Dataset Size", min_value=5, max_value=100, value=20, step=5)
    num_words = st.sidebar.slider("Number of Words per Topic", 1, 20, 5)

    # Load and display dataset info
    st.write("Using the 20 Newsgroups dataset for analysis.")

    if user_input_text:
        documents = utils.preprocess_text(user_input_text)
    elif use_cache:
        st.write("Using cached data")
        documents = load_cached_data(subset_size=dataset_size)
    else:
        st.write("Using new data")
        documents = load_new_data(subset_size=dataset_size)

    # Analyze button
        
    def display_autotune_results(lda_model, num_topics, dictionary, coherence_plot):
        st.write(f"Best Model with {num_topics} Topics")
        topics = lda.get_lda_topics(lda_model, dictionary, num_words)
        display_lda_results(topics)
        st.pyplot(coherence_plot)

    if st.button("Convergence Analysis"):
        if documents:
            convergence_results = lda.convergence_lda_model(documents, start_topics, end_topics, topics_step, start_words, end_words, words_step)
            
            # Plotting Coherence Scores
            # You can create a plot to show how coherence scores vary with different numbers of topics and words

            # Displaying Topics and Words for each step
            for result in convergence_results:
                st.write(f"Number of Topics: {result['num_topics']}, Number of Words: {result['num_words']}")
                display_lda_results(result['topics'])
    # if st.button("Auto-tune LDA"):
    #     if documents:
    #         best_lda_model, best_num_topics, dictionary, coherence_plot = lda.autotune_lda_model(documents)
    #         display_autotune_results(best_lda_model, best_num_topics, dictionary, coherence_plot)



    if st.button("Analyze"):
        if documents:
    
            lda_model, dictionary, coherence_score, perplexity_score = lda.train_lda_model(documents, num_topics)

            # Tokenize and preprocess the original documents again before creating the corpus
            tokenized_documents = [utils.preprocess_text(doc) for doc in documents]

            # Calculate and display the coherence score for LDA
        
            st.write("LDA Topic Coherence Score:", coherence_score)
            st.write("Perplexity Score:", perplexity_score)

            topics = lda.get_lda_topics(lda_model, dictionary, num_words)
            # Display LDA Topics in Tabular Format
            display_lda_results(topics)
            history.store_run_results("LDA", topics, coherence_score, documents)


            

if __name__ == "__main__":
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'main'

    main()