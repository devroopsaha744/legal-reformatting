import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF text extraction
from openai import OpenAI

load_dotenv()

class LegalRewriter:
    def __init__(self, pdf_path, model="gpt-5-mini-2025-08-07"):
        self.model = model
        # Extract vocabulary ONCE and store in memory (preprocessed)
        self.vocab_text = self._extract_vocab(pdf_path)
        # Cache clients to avoid recreation
        self._clients = {}

        # Store the full system prompt with ALL vocabulary in memory
        self.system_prompt = f"""
        Below is a set of legal vocabulary and phrases extracted from a legal reference document.

        Legal words Vocabulary:
        {self.vocab_text}

        Instructions:
        - redraft the following paragraph with the legal Vocabulary attached
        """

    def _get_client(self):
        """Get the appropriate client based on the model, with caching."""
        is_gpt = self.model.startswith("gpt-")
        
        if is_gpt not in self._clients:
            if is_gpt:
                # Use OpenAI client for GPT models
                self._clients[is_gpt] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                # Use Groq client for other models
                self._clients[is_gpt] = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
        
        return self._clients[is_gpt]

    def set_model(self, model):
        """Change the model and update the client."""
        self.model = model

    def _extract_vocab(self, pdf_path):
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()

    def rewrite(self, paragraph: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": paragraph}
            ]
         )
        return response.choices[0].message.content.strip()
    
    def rewrite_stream(self, paragraph: str):
        """Stream the rewritten text token-by-token."""
        client = self._get_client()
        
        # Send ALL vocabulary - it's stored in memory, sent to API with caching
        stream = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": paragraph}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content



