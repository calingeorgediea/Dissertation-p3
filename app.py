import streamlit as st
from topic_models import lsa  # Importing LSA from the topic_models package
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from topic_models import lda
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit.components.v1 as components
import random
nltk.download('punkt')
nltk.download('stopwords')


@st.cache_data
def load_cached_data(subset_size=20):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data

    if subset_size is not None and subset_size < len(data):
        return random.sample(data, subset_size)
    return data

def load_new_data(subset_size=20):
    st.cache_data.clear()
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    if subset_size is not None and subset_size < len(data):
        return random.sample(data, subset_size)
    return data

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

def display_lda_visualization(lda_model, corpus, dictionary):
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

    # Sidebar settings
    st.sidebar.title("Settings")
    selected_model = st.sidebar.selectbox("Select a Topic Modeling Algorithm", ["LSA", "LDA"])
        # In your sidebar settings in the main function
    use_cache = st.sidebar.checkbox("Cache Documents", value=True)

    matrix_type = st.sidebar.selectbox("Select Matrix Type", ["raw", "tfidf"])
    dataset_size = st.sidebar.slider("Select Dataset Size", min_value=5, max_value=100, value=20, step=5)
    # Hyperparameters for LSA
    num_topics = st.sidebar.slider("Number of Topics", 1, 20, 5)
    num_words = st.sidebar.slider("Number of Words per Topic", 1, 20, 5)

    # Load and display dataset info
    st.write("Using the 20 Newsgroups dataset for analysis.")
    if use_cache:
        documents = load_cached_data(subset_size=dataset_size)
    else:
        documents = load_new_data(subset_size=dataset_size)
    
    # Analyze button
    if st.button("Analyze"):
        if documents:
            if selected_model == "LSA":
                # Performing LSA and getting intermediate results
                topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity = lsa.perform_lsa(documents, num_topics, num_words, matrix_type)

                display_results(topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity)

            if selected_model == "LDA":
                preprocessed_documents = preprocess_documents(documents)
                lda_model, dictionary = lda.train_lda_model(preprocessed_documents, num_topics)
                corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]
                topics = lda.get_lda_topics(lda_model, dictionary, num_words)
                # Display LDA Topics in Tabular Format
                display_lda_results(topics)
                # Optional: Display Bar Charts for Each Topic
                display_lda_topic_charts(topics)
                
            else:
                st.warning("Data could not be loaded. Please check the dataset.")

    # New view for stored results
    st.sidebar.title("Stored Results")
    selected_result = st.sidebar.selectbox("Select a Result to View", list(st.session_state['results'].keys()))

 

if __name__ == "__main__":
    main()
