import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load API Key
try:
    with open("API_key.txt", "r") as f:
        api_key = f.read().strip()
except Exception as e:
    st.error("API Key Error! Please check your API key file.")
    st.stop()

# Initialize Gemini 1.5 Pro Model
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key= api_key)


def conversational_tutor(user_input):
    memory.save_context({"human": user_input}, {"ai": "Processing..."})
    
    conversation_history = memory.load_memory_variables({}).get("chat_history", [])
    
    prompt = f"""
    You are a data science tutor. Answer ONLY data science-related questions.
    If the user asks something unrelated, politely decline.
    Keep the conversation aware using memory.
    
    Conversation history: {conversation_history}
    
    User: {user_input}
    """
    
    response = chat_model.invoke(prompt)
    
    # Extract content properly
    if isinstance(response, dict):
        response_text = response.get("content", "I'm sorry, I couldn't generate a response.")
    elif hasattr(response, 'content'):  
        response_text = response.content  
    else:
        response_text = str(response)  

    # Store cleaned response in memory
    memory.save_context({"human": user_input}, {"ai": response_text})  

    return response_text

def main():
    st.set_page_config(page_title="AI Data Science Tutor", page_icon="🎓", layout="wide")
    
    st.sidebar.title("Chat History")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if st.sidebar.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.session_state["messages"] = []
        memory.clear()
        st.rerun()
    
    for chat in st.session_state["chat_history"][-10:]:
        st.sidebar.write(chat)
    
    if st.sidebar.button("New Chat"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        memory.clear()
        st.rerun()
    
    st.title('Conversational AI Data Science Assistant')
    # st.write("Ask me anything about Data Science!")
    st.markdown(
    """
    <div style="display: flex; justify-content: center; gap: 20px;">
        <img src="https://us.123rf.com/450wm/vikvector/vikvector2404/vikvector240400286/228892343-cute-cartoon-robot-with-phone-character-cartoon-style-vector-illustration-eps-10.jpg?ver=6" width="150">
    </div>
    """,unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        response = conversational_tutor(user_input)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.session_state["chat_history"].append(f"{user_input[:50]}... - {response[:50]}...")
        
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()
