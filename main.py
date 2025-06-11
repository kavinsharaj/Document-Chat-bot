import streamlit as st
import json
from datetime import datetime
from ingest import create_vector_store
from app import run_agent

st.set_page_config(page_title="ESG Chatbot", page_icon="🤖")

st.title("📄 ESG Document QA Chatbot")

if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []
if "eval_log_file" not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.eval_log_file = f"eval_log_{timestamp}.json"
uploaded_file = st.file_uploader("Upload ESG Policy PDF", type="pdf")
if uploaded_file and "vectorstore_ready" not in st.session_state:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    create_vector_store("temp.pdf")
    st.session_state.vectorstore_ready = True
    st.session_state.chat_history = []
    st.success("Document processed and vector store created.")
if st.session_state.get("vectorstore_ready", False):
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_input = st.chat_input("Ask a question:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response, scores = run_agent(user_input)

        response_msg = f"{response}\n\n*Confidence: {scores['confidence']:.2f}*"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_msg}
        )

        with st.chat_message("assistant"):
            st.markdown(response_msg)
        st.session_state.eval_logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "question": user_input,
                "answer": response,
                "scores": scores,
            }
        )
        with open(st.session_state.eval_log_file, "w") as f:
            json.dump(st.session_state.eval_logs, f, indent=2)
