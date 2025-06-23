# Simple Chatbot with Hugging Face & Streamlit

A conversational AI chatbot built using open-source LLMs from Hugging Face, with both Jupyter notebook and web interface implementations.

## Features

- **Open-source LLM**: Uses Facebook's BlenderBot-400M model
- **Conversation Memory**: Maintains chat history for context
- **Web Interface**: Clean Streamlit UI for easy interaction
- **Jupyter Support**: Development notebook included

## Files

- `chatbot_app.py` - Streamlit web interface
- `simple_chatbot.ipynb` - Jupyter notebook with step-by-step implementation

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install required dependencies:
```bash
pip install transformers streamlit torch
```

## Usage

### Web Interface
Run the Streamlit app:
```bash
streamlit run chatbot_app.py
```

### Jupyter Notebook
Open and run `simple_chatbot.ipynb` for step-by-step implementation.

## Model

This chatbot uses the `facebook/blenderbot-400M-distill` model, which is:
- Lightweight (400M parameters)
- Optimized for conversational AI
- Downloads automatically on first run

## Requirements

- Python 3.7+
- transformers
- streamlit
- torch

## License

MIT License
