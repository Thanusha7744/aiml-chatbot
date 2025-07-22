import streamlit as st
from chatbot import get_response

st.set_page_config(page_title="AIML FAQ Chatbot", page_icon="🤖")

st.title("🤖 AIML FAQ Chatbot")
st.write("Ask me anything about AI & ML concepts!")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Input box
user_input = st.text_input("You:", "")

if user_input:
    # Save user message
    st.session_state["messages"].append(("You", user_input))

    # Get bot response
    bot_response = get_response(user_input)
    st.session_state["messages"].append(("Bot", bot_response))

# Display chat history
for sender, message in st.session_state["messages"]:
    if sender == "You":
        st.markdown(f"*You:* {message}")
    else:
        st.markdown(f"🤖 Bot:** {message}")