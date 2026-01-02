import streamlit as st

st.title("YouTube Chatbot")

url = st.text_input("Enter YouTube Video URL")

if st.button("Start Conversation"):
    if url:
        st.session_state["video_url"] = url
        st.switch_page("pages/Conversation.py")
    else:
        st.warning("Please enter a valid YouTube URL")

