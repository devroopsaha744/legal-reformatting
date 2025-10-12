import streamlit as st
from backend import LegalRewriter

# Initialize the rewriter with the PDF. Vocabulary is loaded internally and will NOT be shown in the UI.
rewriter = LegalRewriter("legal word ver 2.pdf")

def rewrite_text_stream(text):
    """Function to stream the rewritten text token-by-token."""
    if not text.strip():
        yield "Please enter some text to reformat."
    # Just yield each token as it comes - Streamlit handles the accumulation
    for token in rewriter.rewrite_stream(text):
        yield token

# Streamlit UI
st.title("Legal Text Rewriter")
st.markdown("Rewrite text using legal vocabulary from the attached reference. The vocabulary will NOT be shown; only the editable system prompt is exposed.")

input_text = st.text_area(
    "Input Text",
    placeholder="Enter text to reformat...",
    height=200
)


# Single editable system prompt (shows only instructions+requirements; vocab is hidden)
st.markdown("### System prompt (editable)")
default_ui_prompt = rewriter.get_ui_prompt() if hasattr(rewriter, "get_ui_prompt") else "Instructions:\n\nredraft the paragraph according to the legal vocabular attached"
ui_prompt = st.text_area(
    "System prompt",
    value=default_ui_prompt,
    height=160,
    help="Editable system prompt (instructions + requirements). The extracted vocabulary is not shown."
)

if st.button("Reformat", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to reformat.")
    else:
        # Set the single UI prompt so backend can combine it with the hidden vocab
        if ui_prompt and ui_prompt.strip():
            rewriter.set_ui_prompt(ui_prompt)
        st.subheader("Reformatted Output (Streaming)")
        output_placeholder = st.empty()
        full_response = ""

        for token in rewrite_text_stream(input_text):
            full_response += token
            output_placeholder.markdown(full_response)