import streamlit as st
from topic_models import lsa  # Importing LSA from the topic_models package
from sklearn.datasets import fetch_20newsgroups

def load_data(subset_size=5):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    if subset_size is not None and subset_size < len(data):
        return data[:subset_size]
    return data

def main():
    
    st.title("Topic Modeling Dashboard")

    # Sidebar for algorithm selection and hyperparameters
    st.sidebar.title("Settings")
    selected_model = st.sidebar.selectbox("Select a Topic Modeling Algorithm", ["LSA"])

    # Matrix type selection
    matrix_type = st.sidebar.selectbox("Select Matrix Type", ["raw", "tfidf"])

    # Hyperparameters for LSA
    if selected_model == "LSA":
        num_topics = st.sidebar.slider("Number of Topics", 1, 20, 5)
        num_words = st.sidebar.slider("Number of Words per Topic", 1, 20, 5)

    # Main content area
    st.write("Welcome to the Topic Modeling Dashboard")
    st.write("Choose an algorithm and adjust the hyperparameters to see different results.")

    # Text input for analysis
    text_data = load_data();

    # Analyze button
    if st.button("Analyze"):
        if text_data:
            # documents = text_data.split("\n")
            documents = load_data();
            if selected_model == "LSA":
                # Performing LSA and getting intermediate results
                topics, doc_term_matrix, explained_variance, topic_term_matrix, sparsity = lsa.perform_lsa(documents, num_topics, num_words, matrix_type)

                # Displaying the intermediate results
                st.write("Document-Term Matrix:")
                st.dataframe(doc_term_matrix)

                # Display matrix information
                st.write("Matrix Information:")
                st.write(f"Matrix Sparsity: {sparsity:.2f}%")
                st.write(f"Matrix Shape: {doc_term_matrix.shape}")


                st.write("Explained Variance by Each Topic:")
                st.bar_chart(explained_variance)

                st.write("Topic-Term Matrix:")
                st.dataframe(topic_term_matrix)

                # Displaying the topics
                st.write("LSA Topics:")
                for topic in topics:
                    st.write(topic)
        else:
            st.warning("Please enter some text data for analysis.")

if __name__ == "__main__":
    main()
