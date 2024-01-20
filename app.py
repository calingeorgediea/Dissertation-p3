import streamlit as st
from topic_models import lsa  # Importing LSA from the topic_models package
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def load_data(subset_size=5):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    if subset_size is not None and subset_size < len(data):
        return data[:subset_size]
    return data

def display_results(topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity):
    st.write("Document-Term Matrix:")
    st.dataframe(doc_term_matrix)
    st.write("Explained Variance by Each Topic:")
    st.bar_chart(explained_variance)
    st.write("Topic-Term Matrix:")
    st.dataframe(topic_term_matrix)
    st.write("LSA Topics:")
    for topic in topics:
        st.write(topic)
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
    selected_model = st.sidebar.selectbox("Select a Topic Modeling Algorithm", ["LSA"])
    matrix_type = st.sidebar.selectbox("Select Matrix Type", ["raw", "tfidf"])

    # Hyperparameters for LSA
    num_topics = st.sidebar.slider("Number of Topics", 1, 20, 5)
    num_words = st.sidebar.slider("Number of Words per Topic", 1, 20, 5)

    # Load and display dataset info
    st.write("Using the 20 Newsgroups dataset for analysis.")
    documents = load_data()

    # Analyze button
    if st.button("Analyze"):
        if documents:
            if selected_model == "LSA":
                # Performing LSA and getting intermediate results
                topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity = lsa.perform_lsa(documents, num_topics, num_words, matrix_type)

                # Update session state with new results
                key = f"{selected_model}-{matrix_type}"
                st.session_state['results'][key] = {
                    'topics': topics,
                    'doc_term_matrix': doc_term_matrix,
                    'explained_variance': explained_variance,
                    'topic_term_matrix': topic_term_matrix,
                    'U_matrix': U_matrix,
                    'Sigma_matrix': Sigma_matrix,
                    'VT_matrix': VT_matrix,
                    'sparsity': sparsity
                }

                # Displaying the current run results
                display_results(topics, doc_term_matrix, explained_variance, topic_term_matrix, U_matrix, Sigma_matrix, VT_matrix, sparsity)

        else:
            st.warning("Data could not be loaded. Please check the dataset.")

    # New view for stored results
    st.sidebar.title("Stored Results")
    selected_result = st.sidebar.selectbox("Select a Result to View", list(st.session_state['results'].keys()))

    if 'results' in st.session_state and selected_result:
        result = st.session_state['results'][selected_result]
        st.write(f"Results for {selected_result}")

        # Display stored results
        display_results(
            result['topics'], 
            result['doc_term_matrix'], 
            result['explained_variance'], 
            result['topic_term_matrix'], 
            result.get('U_matrix'), 
            result.get('Sigma_matrix'), 
            result.get('VT_matrix'), 
            result['sparsity']
        )

if __name__ == "__main__":
    main()
