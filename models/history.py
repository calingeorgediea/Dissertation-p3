import streamlit as st
import datetime

def store_run_results(algorithm, topics, coherence_score, other_metrics = "", documents = ""):
    """
    Store the results of a single run in the Streamlit session state.
    """
    if 'run_history' not in st.session_state:
        st.session_state.run_history = []

    run_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.run_history.append({
        "run_id": run_id,
        "algorithm": algorithm,
        "topics": topics,
        "coherence_score": coherence_score,
        "documents": documents  # Be cautious with large data sets
    })

def get_recent_runs(limit=10):
    """
    Retrieve the most recent runs from the Streamlit session state.
    """
    if 'run_history' in st.session_state:
        return st.session_state.run_history[-limit:]
    return []
