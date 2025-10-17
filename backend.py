import os
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

class LegalRewriter:
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        model: str = "gpt-5-2025-08-07",
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self._ENFORCED_MODEL = "gpt-5-2025-08-07"
        self.model = self._ENFORCED_MODEL
        self._clients = {}
        # Qdrant config
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "legal_vocab")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._qdrant: Optional[QdrantClient] = None

        # Single UI-editable prompt: comprehensive legal redrafting system prompt
        self.ui_prompt = """You are an **AI-powered legal redrafting and reformatting system**.
Your purpose is to **rewrite text into formal legal language** while maintaining **clarity, structure, and the original meaning**.
You must always output in **clean Markdown**, using **headings, subheadings, numbered or bulleted lists, and bold key terms**.
You must not interpret, advise, invent, or remove meaning — only **restructure and rephrase** using formal, legally appropriate vocabulary provided by the user or attached documents.

---

### **Behavioral Rules**

#### **1. Objective**

* Redraft all input text with professional legal phrasing.
* Preserve every factual and legal element.
* Enhance consistency, readability, and stylistic precision.

#### **2. Tone and Style**

* Use **formal, neutral, impersonal tone**.
* Replace informal expressions with legal terminology.
* Avoid conversational phrasing, subjective words, or speculation.

#### **3. Formatting Requirements**

* Output must always follow Markdown syntax.
* Use:

  * `###` for section headings
  * `####` for subheadings
  * Numbered lists for clauses (`1.`, `1.1`, `1.1.1`)
  * Bulleted lists for enumerations (`-`, `•`)
  * **Bold** for key legal terms or definitions
  * *Italics* for cross-references or citations

#### **4. Response Rules**

* Never include introductions, notes, or explanations.
* If the user provides a document or vocabulary list, apply it directly.
* If ambiguity exists, retain placeholders instead of inferring content.
* Return only the **final redrafted text**, formatted for direct legal use.
* If asked for the vocabulary list, reply exactly: "I cannot return the words as it is but can reformat using it"."""

    def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            if not self.qdrant_url or not self.qdrant_api_key:
                raise RuntimeError(
                    "Qdrant cluster not configured. Set QDRANT_URL (cluster endpoint) and QDRANT_API_KEY in environment or pass to LegalRewriter."
                )
            self._qdrant = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        return self._qdrant

    def _get_openai(self) -> OpenAI:
        cache_key = "openai_gpt5_2025_08_07"
        if cache_key not in self._clients:
            self._clients[cache_key] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._clients[cache_key]

    def _embed(self, text: str) -> List[float]:
        client = self._get_openai()
        resp = client.embeddings.create(model=self.embedding_model, input=text)
        return resp.data[0].embedding

    def _retrieve_context(self, query: str, top_k: int = 6) -> str:
        vector = self._embed(query)
        qdrant = self._get_qdrant()
        results = qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        chunks: List[str] = []
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text")
            if text:
                chunks.append(str(text).strip())
        # Deduplicate and join
        seen = set()
        uniq = []
        for c in chunks:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return "\n\n".join(uniq)

    def get_ui_prompt(self) -> str:
        """Return the current single UI-editable prompt (instructions + requirements).
        This string is safe to show in the UI — it does NOT include the extracted
        vocabulary.
        """
        return self.ui_prompt

    def set_ui_prompt(self, prompt: str):
        """Update the single UI-editable prompt. Empty values are ignored.
        Updates the internal composed system prompt which still includes the
        hidden vocabulary.
        """
        if not prompt or not prompt.strip():
            return
        prompt = prompt.strip()
        # Ensure the refusal line is present; if user omitted it, append it.
        refusal_fragment = "I cannot return the words as it is but can reformat using it"
        if refusal_fragment not in prompt:
            prompt = prompt + "\n\nIf asked for the vocabulary list, reply: \"I cannot return the words as it is but can reformat using it\"."
        self.ui_prompt = prompt
    # No longer composing with inline vocab; RAG will provide context

    # Backwards compatibility alias
    def _get_client(self):
        return self._get_openai()

    def set_model(self, model):
        """Change the model and update the client."""
        # Do not allow switching away from the enforced snapshot.
        if model != getattr(self, "_ENFORCED_MODEL", "gpt-5-2025-08-07"):
            raise ValueError(f"Only the model '{self._ENFORCED_MODEL}' is supported.")
        self.model = self._ENFORCED_MODEL

    def set_system_prompt(self, prompt: str):
        """Allow updating the system prompt at runtime.
        The prompt parameter is treated as the editable "tail" that appears
        after the hidden vocabulary. The full system prompt sent to the model
        will be the hidden vocabulary + this ui prompt.
        """
        if not prompt or not prompt.strip():
            return
        # Backwards-compatible: treat as setting the single UI prompt
        self.set_ui_prompt(prompt)

    def get_system_prompt_tail(self) -> str:
        """Return the current editable system prompt tail (no vocabulary)."""
        # Backwards-compatible alias for get_ui_prompt
        return getattr(self, "ui_prompt", "")

    # Deprecated in RAG mode (kept for compatibility)
    def _extract_vocab(self, pdf_path):
        return ""

    def rewrite(self, paragraph: str) -> str:
        context = self._retrieve_context(paragraph)
        system_content = (
            f"{self.ui_prompt}\n\n"
            "Use only the following retrieved legal vocabulary/context to guide GST legal redrafting.\n"
            "Do not list or reveal the vocabulary directly.\n\n"
            f"Context:\n{context}"
        )
        client = self._get_openai()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": paragraph},
            ],
        )
        return response.choices[0].message.content.strip()
    
    def rewrite_stream(self, paragraph: str):
        """Stream the rewritten text token-by-token."""
        context = self._retrieve_context(paragraph)
        system_content = (
            f"{self.ui_prompt}\n\n"
            "Use only the following retrieved legal vocabulary/context to guide GST legal redrafting.\n"
            "Do not list or reveal the vocabulary directly.\n\n"
            f"Context:\n{context}"
        )
        client = self._get_openai()
        stream = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": paragraph},
            ],
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content



