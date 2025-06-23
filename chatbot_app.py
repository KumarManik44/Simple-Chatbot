import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load model and tokenizer (cached)
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Generate response function
def generate_response(user_input, model, tokenizer, conversation_history):
    # Use recent history to avoid context confusion
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    
    # Tokenize input
    inputs = tokenizer.encode_plus(user_input, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

# Main app
def main():
    st.title("🤖 AI Chatbot")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, tokenizer = load_model()
    
    # Chat interface
    st.subheader("Chat with the AI")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    prompt, 
                    model, 
                    tokenizer, 
                    st.session_state.conversation_history
                )
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update conversation history
        st.session_state.conversation_history.append(prompt)
        st.session_state.conversation_history.append(response)
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Model:** facebook/blenderbot-400M-distill")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

if __name__ == "__main__":
    main()
