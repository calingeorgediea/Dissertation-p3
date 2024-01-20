from gensim import corpora, models

def train_lda_model(documents, num_topics=5):
    # Create a dictionary from the documents
    dictionary = corpora.Dictionary(documents)

    # Create a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]

    # Train the LDA model
    lda_model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)

    return lda_model, dictionary

def get_lda_topics(lda_model, dictionary, num_words=5):
    return [lda_model.show_topic(topicid, num_words) for topicid in range(lda_model.num_topics)]
