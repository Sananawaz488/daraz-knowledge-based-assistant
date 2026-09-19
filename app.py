"""
Daraz Customer Support Operations Assistant
--------------------------------------------
A Streamlit chat app that answers customer-support questions using a
PRE-BUILT FAISS index (created separately by ingest.py).

This app NEVER re-embeds or reprocesses PDFs. It only:
  1. Loads an existing FAISS index + metadata.json from disk
  2. Embeds the user's QUESTION at query time (not the documents)
  3. Retrieves the most relevant chunks (optionally restricted to one
     knowledge-base section, e.g. Returns, Delivery, Refunds, Seller)
  4. Sends those chunks + the question to a Groq-hosted LLM
     (openai/gpt-oss-120b) to generate a grounded answer

Folder expected next to this file:
    faiss_index/
        index.faiss
        metadata.json
"""

import json
import os

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    from groq import Groq
except ImportError:
    Groq = None


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

RETRIEVE_K = 15   # candidates pulled from FAISS before section filtering
TOP_K = 5         # chunks actually sent to the LLM as context

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#0F1111"

SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant.
Answer the user's question using ONLY the information in the provided
knowledge base excerpts below. These excerpts come from official Daraz
policy documents.

Rules:
- If the excerpts do not contain enough information to answer, say so
  clearly and suggest the customer contact Daraz Customer Support directly.
- Be concise, polite, and professional - written for a support agent or
  a customer, whichever fits the question.
- Do not invent policy details that are not present in the excerpts.
- When useful, mention which section (e.g. Returns, Delivery, Refund,
  Seller) the answer relates to.
"""


# ────────────────────────────────────────────────────────────────
# Cached loaders — index and models are loaded once per session
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_index_and_metadata(index_dir: str):
    index_path = os.path.join(index_dir, "index.faiss")
    metadata_path = os.path.join(index_dir, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ────────────────────────────────────────────────────────────────
# Retrieval
# ────────────────────────────────────────────────────────────────
def get_sections(metadata):
    sections = sorted({entry.get("department", "General") for entry in metadata})
    return sections


def search(query, index, metadata, model, section=None, top_k=TOP_K, retrieve_k=RETRIEVE_K):
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(query_embedding, retrieve_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        entry = metadata[idx]
        if section and section != "All Sections" and entry.get("department") != section:
            continue
        results.append({
            "score": float(score),
            "department": entry.get("department", "General"),
            "source_file": entry.get("source_file", "unknown"),
            "text": entry.get("text", ""),
        })
        if len(results) >= top_k:
            break

    return results


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[Excerpt {i} | Section: {c['department']} | Source: {c['source_file']}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(client, question, context, history):
    if client is None:
        return (
            "⚠️ The assistant isn't fully configured yet — no Groq API key was "
            "found. Please ask an administrator to set `GROQ_API_KEY` in "
            "Streamlit secrets."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # include a little recent chat history for conversational context
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"Knowledge base excerpts:\n\n{context}\n\n"
        f"Customer question: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Sorry, something went wrong while generating a response: {e}"


# ────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────
def inject_branding():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #FAFAFA;
            }}
            [data-testid="stSidebar"] {{
                background-color: {DARAZ_DARK};
            }}
            [data-testid="stSidebar"] * {{
                color: #FFFFFF !important;
            }}
            .daraz-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 14px 20px;
                background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
                border-radius: 10px;
                margin-bottom: 18px;
            }}
            .daraz-header h1 {{
                color: white;
                font-size: 22px;
                margin: 0;
                font-weight: 700;
            }}
            .daraz-header p {{
                color: #FFF3EC;
                margin: 0;
                font-size: 13px;
            }}
            .stChatMessage {{
                border-radius: 12px;
            }}
            div[data-testid="stChatInput"] textarea {{
                border: 1px solid {DARAZ_ORANGE} !important;
            }}
            .source-tag {{
                display: inline-block;
                background-color: #FFF0E6;
                color: {DARAZ_ORANGE};
                border: 1px solid {DARAZ_ORANGE};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 12px;
                margin-right: 6px;
                margin-top: 4px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="daraz-header">
            <div style="font-size:28px;">🛍️</div>
            <div>
                <h1>Daraz Support Assistant</h1>
                <p>Customer Support Operations · Knowledge Base Q&amp;A</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata):
    st.sidebar.markdown("### 📚 Knowledge Base")
    st.sidebar.caption("Restrict your search to one section, or search all.")

    sections = get_sections(metadata)
    options = ["All Sections"] + sections

    selected = st.sidebar.radio(
        "Section",
        options=options,
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.caption(
        "This assistant answers from Daraz's official policy documents "
        "(Returns, Delivery, Refunds, Seller guidelines). It does not "
        "make up policy details outside these documents."
    )

    if st.sidebar.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    return selected


def main():
    st.set_page_config(
        page_title="Daraz Support Assistant",
        page_icon="🛍️",
        layout="centered",
    )
    inject_branding()

    index, metadata = load_index_and_metadata(INDEX_DIR)

    if index is None or metadata is None:
        st.error(
            f"Couldn't find a pre-built FAISS index at `{INDEX_DIR}/`.\n\n"
            "This app only loads an existing index — it does not build one. "
            "Run `ingest.py` first to create `index.faiss` and `metadata.json`, "
            "then place the `faiss_index/` folder next to `app.py`."
        )
        st.stop()

    if Groq is None:
        st.warning(
            "The `groq` package isn't installed. Add `groq` to requirements.txt "
            "and reinstall dependencies."
        )

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    groq_client = get_groq_client()

    if groq_client is None:
        st.sidebar.warning(
            "⚠️ No API key configured. Add `GROQ_API_KEY` to your "
            "Streamlit secrets to enable answer generation."
        )

    selected_section = render_sidebar(metadata)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                tags = "".join(
                    f'<span class="source-tag">{s}</span>' for s in msg["sources"]
                )
                st.markdown(tags, unsafe_allow_html=True)

    placeholder = "Ask about returns, delivery, refunds, or seller policies..."
    user_question = st.chat_input(placeholder)

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                chunks = search(
                    user_question,
                    index,
                    metadata,
                    embedding_model,
                    section=selected_section,
                )

            if not chunks:
                answer = (
                    "I couldn't find anything relevant in "
                    f"**{selected_section}** for that question. Try selecting "
                    "**All Sections**, or rephrasing your question."
                )
                sources = []
            else:
                context = build_context(chunks)
                with st.spinner("Generating answer..."):
                    answer = generate_answer(
                        groq_client, user_question, context, st.session_state.messages
                    )
                sources = sorted({f"{c['department']} · {c['source_file']}" for c in chunks})

            st.markdown(answer)
            if sources:
                tags = "".join(f'<span class="source-tag">{s}</span>' for s in sources)
                st.markdown(tags, unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
