import streamlit as st
from chat import Chat
from dotenv import load_dotenv

load_dotenv()

st.title("🦜 Sarah Chat Assistant")


@st.cache_resource
def load_model():
    return Chat()


# Load LLM Model
chat_model = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(chat_model.generate_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
