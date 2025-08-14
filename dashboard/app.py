# dashboard/app.py

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Backend API Configuration ---
BACKEND_URL = "http://backend:8000/query"

# --- UI Components ---
st.title("📄 AI Document Chatbot")
st.caption("Powered by a local RAG pipeline with `deepset/roberta-base-squad2`")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist for an assistant message
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(f"Source: {source.get('source', 'Unknown')}")
                    st.text(source.get('content', 'No content available.'))


# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Call Backend API ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Prepare the request payload
            # MODIFIED: Reduced top_k from 3 to 2 to prevent context overflow
            payload = {"query": prompt, "top_k": 2}
            
            # Send POST request to the backend
            response = requests.post(BACKEND_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            result = response.json()
            answer = result.get("answer", "No answer found.")
            sources = result.get("sources", [])
            
            # Display the answer
            message_placeholder.markdown(answer)
            
            # Display sources in an expander
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.info(f"Source: {source.get('source', 'Unknown')}")
                        st.text(source.get('content', 'No content available.'))
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })

        except requests.exceptions.RequestException as e:
            error_message = f"Could not connect to the backend API. Please ensure the backend is running. Error: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

