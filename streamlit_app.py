import streamlit as st
from backend import LegalRewriter

# Initialize the rewriter with the PDF
rewriter = LegalRewriter("legal word.pdf")

# Available models
MODELS = {
    "GPT-5 Mini (2025-08-07)": "gpt-5-mini-2025-08-07",
    "Groq GPT-OSS": "openai/gpt-oss-120b",
    "GPT-5 (2025-08-07)": "gpt-5-2025-08-07"
}

def rewrite_text_stream(text, model):
    """Function to stream the rewritten text token-by-token."""
    if not text.strip():
        yield "Please enter some text to reformat."

    # Update the model in the rewriter
    rewriter.set_model(MODELS[model])

    # Just yield each token as it comes - Streamlit handles the accumulation
    for token in rewriter.rewrite_stream(text):
        yield token

# Streamlit UI
st.title("Legal Text Rewriter")
st.markdown("Rewrite text using legal vocabulary from the reference document. Watch the output stream in real-time!")

model = st.selectbox(
    "Select Model",
    options=list(MODELS.keys()),
    index=0,
    help="Choose between OpenAI GPT models or Groq models"
)

input_text = st.text_area(
    "Input Text",
    placeholder="Enter text to reformat...",
    height=200
)

if st.button("Reformat", type="primary"):
    if input_text.strip():
        st.subheader("Reformatted Output (Streaming)")
        output_placeholder = st.empty()
        full_response = ""
        
        for token in rewrite_text_stream(input_text, model):
            full_response += token
            output_placeholder.markdown(full_response)
    else:
        st.warning("Please enter some text to reformat.")