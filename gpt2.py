# gpt2.py

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import sent_tokenize, word_tokenize


def lemmatize_stemming(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text, pos='v')

def preprocess_text(text):
    result = []
    for sentence in sent_tokenize(text):
        processed_sentence = []
        for token in gensim.utils.simple_preprocess(sentence):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                processed_sentence.append(lemmatize_stemming(token))
        if processed_sentence:
            result.append(' '.join(processed_sentence))
    return result

def load_new_data(subset_size=20):
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    if subset_size is not None and subset_size < len(data):
        data = random.sample(data, subset_size)

    processed_data = [preprocess_text(doc) for doc in data]
    return processed_data

def display_gpt2_page():
    model_name = "gpt2"  # You can choose "gpt2", "gpt2-medium", "gpt2-large", etc.
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    st.title("Text Enrichment with GPT-2")

    # Initialize session state for selected sentence
    if 'selected_sentence' not in st.session_state:
        st.session_state['selected_sentence'] = None

    # Load and preprocess new data
    data = load_new_data(1)  # Load one document for demonstration

    # If there are sentences in the data, display them in a select box
    if data:
        sentences = data[0]  # Assuming data[0] is a list of sentences
        # Update the session state when a new sentence is selected
        
        generate_button = st.button("Enrich Sentence")

        selected_sentence = st.selectbox("Select a sentence to enrich", sentences, key="sentence_selector")
                

        if generate_button and selected_sentence:
            # Encode and generate text
            inputs = tokenizer.encode(selected_sentence, return_tensors="pt")
            outputs = model.generate(inputs, max_length=40, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.subheader("Enriched Sentence")
            st.write(generated_text)
        else:
            st.write("No data available for processing.")
