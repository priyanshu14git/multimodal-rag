"""
MultiRAG — Multimodal Document Intelligence
Streamlit frontend for the existing multimodal RAG backend.

This file only does presentation + HTTP calls. All retrieval/generation
logic stays in the FastAPI backend (backend/app/services/*). The three
endpoints used are the ones that already exist:

    GET  /health
    POST /ingest
    POST /ask

/ask now additionally accepts optional fields (top_k_text, top_k_images,
include_text, include_images, include_tables, temperature) so the
retrieval/generation controls in this UI are real, not decorative — see
backend/app/models/schemas.py::AskRequest. Every field has a default that
reproduces the pipeline's original fixed behavior, so any other caller of
/ask that doesn't send them is unaffected.
"""

import html
import inspect
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from utils import parse_markdown_table

# =============================================================================
# Config
# =============================================================================

API_URL = "http://127.0.0.1:8000"

# Ollama exposes its own REST API on this host/port. We query it directly
# for a real "is the model server up" check — this is Ollama's existing
# endpoint, not something invented on our FastAPI backend. Change this if
# your Ollama instance runs elsewhere (see backend/app/core/config.py for
# the same default used server-side).
OLLAMA_HOST = "http://127.0.0.1:11434"

EXAMPLE_PROMPTS = [
    "Explain the architecture shown in the document.",
    "Summarize the main findings.",
    "What does Table 3 show?",
    "Compare the methods described in the paper.",
]

_TYPE_ICON = {"text": "📄", "table": "📊", "image": "🖼️"}

st.set_page_config(
    page_title="MultiRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Feature-detect newer widgets so this runs across Streamlit versions
# without throwing on an unsupported kwarg.
_HAS_BORDER_CONTAINER = "border" in inspect.signature(st.container).parameters
_HAS_TOGGLE = hasattr(st, "toggle")
_HAS_POPOVER = hasattr(st, "popover")


def card():
    """A bordered container where supported, a plain one otherwise."""
    if _HAS_BORDER_CONTAINER:
        return st.container(border=True)
    return st.container()


def toggle(label: str, value: bool, key: str) -> bool:
    """st.toggle on newer Streamlit, a checkbox fallback on older installs."""
    if _HAS_TOGGLE:
        return st.toggle(label, value=value, key=key)
    return st.checkbox(label, value=value, key=key)


# =============================================================================
# Theme
# =============================================================================

_PALETTES = {
    "dark": dict(
        bg="#0E0F11", surface="#16181B", surface2="#1D2024", border="#2A2D31",
        text="#E7E8EA", text_dim="#9AA0A6", accent="#6C8EF5",
        good="#35C48F", warn="#E8A33D", bad="#E8555A",
    ),
    "light": dict(
        bg="#FAFAFA", surface="#FFFFFF", surface2="#F2F3F5", border="#E4E5E8",
        text="#1A1B1E", text_dim="#6B7280", accent="#4C6EF5",
        good="#1F9D6E", warn="#B4780A", bad="#C13A3F",
    ),
}


def inject_css(theme: str) -> None:
    p = _PALETTES[theme]
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            background: {p['bg']};
            color: {p['text']};
        }}
        section[data-testid="stSidebar"] {{
            background: {p['surface']};
            border-right: 1px solid {p['border']};
        }}
        .urag-user-row {{
            display: flex;
            justify-content: flex-end;
            margin: 4px 0 16px 0;
        }}
        .urag-user-bubble {{
            background: {p['surface2']};
            border: 1px solid {p['border']};
            border-radius: 10px;
            padding: 9px 15px;
            max-width: 72%;
            font-size: 0.95rem;
            line-height: 1.45;
            white-space: pre-wrap;
        }}
        .urag-assistant-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: {p['text_dim']};
            font-size: 0.85rem;
            margin: 2px 0 6px 0;
            font-weight: 500;
        }}
        .urag-badge {{
            display: inline-block;
            border: 1px solid {p['border']};
            background: {p['surface2']};
            color: {p['text_dim']};
            border-radius: 6px;
            padding: 1px 8px;
            font-size: 0.73rem;
            margin-left: 2px;
        }}
        .urag-welcome {{
            text-align: center;
            padding: 56px 12px 20px 12px;
        }}
        .urag-welcome h2 {{
            margin: 10px 0 4px 0;
            font-weight: 600;
        }}
        .urag-welcome p {{
            color: {p['text_dim']};
            margin: 0 0 22px 0;
        }}
        .urag-not-grounded {{
            border-left: 3px solid {p['warn']};
            background: {p['surface2']};
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            color: {p['text_dim']};
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Session state
# =============================================================================

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("doc_id", None)
    ss.setdefault("document_name", None)
    ss.setdefault("doc_stats", {})
    ss.setdefault("messages", [])
    ss.setdefault("uploader_key", 0)
    ss.setdefault(
        "settings",
        {
            "theme": "dark",
            "top_k": 5,
            "include_text": True,
            "include_images": True,
            "include_tables": True,
            "show_evidence": False,
            "temperature": 0.15,
        },
    )


# =============================================================================
# Backend / Ollama health
# =============================================================================

def check_backend_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_ollama_health() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# =============================================================================
# Header
# =============================================================================

def render_header() -> None:
    col_title, col_status, col_settings, col_theme = st.columns([4, 3, 1, 1])

    with col_title:
        st.markdown("## 🧠 MultiRAG")
        st.caption("Multimodal Document Intelligence")

    with col_status:
        backend_ok = check_backend_health()
        ollama_ok = check_ollama_health()
        b_dot = "🟢" if backend_ok else "🔴"
        o_dot = "🟢" if ollama_ok else "🔴"
        st.markdown(
            f"{b_dot} Backend&nbsp;&nbsp;&nbsp;{o_dot} Ollama",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{'Connected' if backend_ok else 'Offline'} · "
            f"{'Available' if ollama_ok else 'Unavailable'}"
        )

    with col_settings:
        render_settings_trigger()

    with col_theme:
        icon = "🌙" if st.session_state.settings["theme"] == "light" else "☀️"
        if st.button(icon, help="Toggle theme", key="theme_toggle_btn"):
            st.session_state.settings["theme"] = (
                "light" if st.session_state.settings["theme"] == "dark" else "dark"
            )
            st.rerun()

    st.divider()


def render_settings_trigger() -> None:
    if _HAS_POPOVER:
        with st.popover("⚙️"):
            render_settings_body()
    else:
        with st.expander("⚙️ Settings"):
            render_settings_body()


def render_settings_body() -> None:
    st.markdown("**Generation**")
    st.session_state.settings["temperature"] = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.settings["temperature"],
        0.05,
        key="temperature_slider",
        help="Lower is more deterministic and literal; higher allows more varied wording.",
    )
    st.caption("Retrieval controls (Top-K, text/image/table toggles) are in the left sidebar.")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("#### Document")

        uploaded = st.file_uploader(
            "Upload document",
            type=["pdf"],
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if uploaded is not None and st.button(
            "Ingest document", use_container_width=True, type="primary"
        ):
            ingest_document(uploaded)

        if st.session_state.doc_id:
            st.markdown(f"📄 **{st.session_state.document_name}**")
            stats = st.session_state.doc_stats
            st.caption(
                f"{stats.get('num_pages', '—')} pages · "
                f"{stats.get('text_chunks', '—')} text chunks · "
                f"{stats.get('tables', '—')} tables · "
                f"{stats.get('images', '—')} images"
            )
            st.success("Document indexed", icon="✅")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("New document", use_container_width=True):
                    reset_document()
                    st.rerun()
            with col_b:
                if st.button("Remove document", use_container_width=True):
                    reset_document()
                    st.rerun()

            if st.button("Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        else:
            st.caption("Upload a PDF and click **Ingest document** to begin.")

        st.divider()
        st.markdown("#### Retrieval")

        s = st.session_state.settings
        s["include_text"] = toggle("Text retrieval", s["include_text"], key="tg_text")
        s["include_images"] = toggle("Image retrieval", s["include_images"], key="tg_images")
        s["include_tables"] = toggle("Table retrieval", s["include_tables"], key="tg_tables")

        st.markdown("**Top K**")
        s["top_k"] = st.slider(
            "Results per query", 1, 10, s["top_k"], key="sl_topk", label_visibility="collapsed"
        )

        st.divider()
        s["show_evidence"] = toggle(
            "Show retrieved evidence", s["show_evidence"], key="tg_evidence"
        )


def ingest_document(uploaded_file) -> None:
    with st.spinner("Indexing document — parsing text, tables and images…"):
        try:
            resp = requests.post(
                f"{API_URL}/ingest",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
                timeout=600,
            )
        except requests.exceptions.RequestException as exc:
            st.error("Backend is unavailable. Is the FastAPI server running?")
            with st.expander("Technical details"):
                st.code(str(exc))
            return

        if resp.status_code != 200:
            st.error("Unable to process this document.")
            with st.expander("Technical details"):
                st.code(resp.text)
            return

        data = resp.json()
        st.session_state.doc_id = data["doc_id"]
        st.session_state.document_name = data["source_filename"]
        st.session_state.doc_stats = {
            "num_pages": data.get("num_pages"),
            "text_chunks": data.get("text_chunks"),
            "tables": data.get("tables"),
            "images": data.get("images"),
        }
        st.session_state.messages = []

    st.rerun()


def reset_document() -> None:
    st.session_state.doc_id = None
    st.session_state.document_name = None
    st.session_state.doc_stats = {}
    st.session_state.messages = []
    st.session_state.uploader_key += 1


# =============================================================================
# Ask flow
# =============================================================================

def ask_question(question: str) -> None:
    if not st.session_state.doc_id:
        st.warning("Please upload a PDF before asking a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    s = st.session_state.settings
    payload = {
        "doc_id": st.session_state.doc_id,
        "question": question,
        "top_k_text": s["top_k"],
        "top_k_images": s["top_k"],
        "include_text": s["include_text"],
        "include_images": s["include_images"],
        "include_tables": s["include_tables"],
        "temperature": s["temperature"],
    }

    with st.status("🔎 Retrieving relevant evidence…", expanded=False) as status:
        try:
            resp = requests.post(f"{API_URL}/ask", json=payload, timeout=600)
        except requests.exceptions.RequestException as exc:
            status.update(label="Backend is unavailable.", state="error")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Backend is unavailable.",
                    "error": str(exc),
                }
            )
            return

        if resp.status_code != 200:
            status.update(label="Unable to process this question.", state="error")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Unable to process this question.",
                    "error": resp.text,
                }
            )
            return

        status.update(label="🧠 Analyzing multimodal context…")
        data = resp.json()
        status.update(label="✍️ Generating response… done", state="complete")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data.get("answer", "No answer returned."),
            "citations": data.get("citations", []),
            "retrieved_text": data.get("retrieved_text", []),
            "retrieved_images": data.get("retrieved_images", []),
            "grounded": data.get("grounded", True),
        }
    )


# =============================================================================
# Chat rendering
# =============================================================================

def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="urag-user-row"><div class="urag-user-bubble">'
            f'{html.escape(msg["content"])}</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="urag-assistant-header">🤖 <span>MultiRAG</span>'
        '<span class="urag-badge">Multimodal answer</span></div>',
        unsafe_allow_html=True,
    )

    if msg.get("error"):
        st.error(msg["content"])
        with st.expander("Technical details"):
            st.code(msg["error"])
        return

    if msg.get("grounded") is False:
        st.markdown(
            '<div class="urag-not-grounded">⚠️ No strongly relevant content was found for '
            'this question in the document — treat this answer with caution.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(msg["content"])

    if msg.get("citations"):
        render_sources(msg)


def render_sources(msg: dict) -> None:
    st.markdown("**Sources**")

    retrieved_text = {item["source_id"]: item for item in msg.get("retrieved_text", [])}
    retrieved_images = {item["source_id"]: item for item in msg.get("retrieved_images", [])}
    citations = msg["citations"]

    n_cols = min(len(citations), 4) or 1
    cols = st.columns(n_cols)

    for i, citation in enumerate(citations):
        sid = citation.get("source_id", "?")
        icon = _TYPE_ICON.get(citation.get("type"), "📄")
        with cols[i % n_cols]:
            with card():
                st.markdown(f"**{icon} Page {citation.get('page', '—')}**")
                st.caption(citation.get("label", ""))
                with st.expander("View"):
                    render_source_detail(sid, retrieved_text, retrieved_images)


def render_source_detail(sid: str, retrieved_text: dict, retrieved_images: dict) -> None:
    if sid in retrieved_text:
        item = retrieved_text[sid]
        if item.get("type") == "table":
            df = parse_markdown_table(item.get("content", ""))
            if df is not None:
                st.dataframe(df, use_container_width=True)
            else:
                st.markdown(item.get("content", ""))
        else:
            st.write(item.get("content", ""))
        if item.get("distance") is not None:
            st.caption(f"Retrieval distance: {item['distance']:.3f}")

    elif sid in retrieved_images:
        item = retrieved_images[sid]
        path = item.get("file_path")
        if path and Path(path).exists():
            st.image(
                path,
                caption=item.get("caption") or f"Page {item.get('page')}",
                use_container_width=True,
            )
        else:
            st.caption(f"Image path: {path}")
        if item.get("distance") is not None:
            st.caption(f"Retrieval distance: {item['distance']:.3f}")

    else:
        st.caption("No additional detail available for this source.")


def render_empty_state() -> None:
    st.markdown(
        '<div class="urag-welcome">'
        '<div style="font-size:2.4rem;">🧠</div>'
        "<h2>Upload a PDF to begin</h2>"
        "<p>Analyze text, tables and images with multimodal RAG.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        '<div class="urag-welcome">'
        '<div style="font-size:2.2rem;">🧠</div>'
        "<h2>Multimodal Document Intelligence</h2>"
        "<p>Ask questions about your PDF using text, tables and images.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"example_{i}", use_container_width=True):
                ask_question(prompt)
                st.rerun()


def render_chat_area() -> None:
    if not st.session_state.doc_id:
        render_empty_state()
        return

    if not st.session_state.messages:
        render_welcome()
        return

    for msg in st.session_state.messages:
        render_message(msg)


# =============================================================================
# Right-side evidence panel
# =============================================================================

def render_retrieved_evidence() -> None:
    st.markdown("#### Retrieved Evidence")

    assistant_msgs = [
        m for m in st.session_state.messages if m["role"] == "assistant" and not m.get("error")
    ]
    if not assistant_msgs:
        st.caption("Ask a question to see retrieved evidence here.")
        return

    latest = assistant_msgs[-1]
    retrieved_text = latest.get("retrieved_text", [])
    retrieved_images = latest.get("retrieved_images", [])
    text_items = [t for t in retrieved_text if t.get("type") != "table"]
    table_items = [t for t in retrieved_text if t.get("type") == "table"]

    tab_text, tab_tables, tab_images = st.tabs(["Text", "Tables", "Images"])

    with tab_text:
        if not text_items:
            st.caption("No text evidence retrieved for the latest question.")
        for item in text_items:
            with card():
                st.markdown(f"**[{item['source_id']}]** Page {item.get('page')}")
                st.write(item.get("content", ""))

    with tab_tables:
        if not table_items:
            st.caption("No table evidence retrieved for the latest question.")
        for item in table_items:
            with card():
                st.markdown(f"**[{item['source_id']}]** Page {item.get('page')}")
                df = parse_markdown_table(item.get("content", ""))
                if df is not None:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.markdown(item.get("content", ""))

    with tab_images:
        if not retrieved_images:
            st.caption("No image evidence retrieved for the latest question.")
        for item in retrieved_images:
            with card():
                st.markdown(f"**[{item['source_id']}]** Page {item.get('page')}")
                path = item.get("file_path")
                if path and Path(path).exists():
                    st.image(path, caption=item.get("caption"), use_container_width=True)
                if item.get("distance") is not None:
                    st.caption(f"Distance: {item['distance']:.3f}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    init_state()
    inject_css(st.session_state.settings["theme"])

    render_header()
    render_sidebar()

    show_evidence = st.session_state.settings["show_evidence"] and bool(
        st.session_state.messages
    )

    if show_evidence:
        col_chat, col_evidence = st.columns([2, 1])
    else:
        col_chat, col_evidence = st.container(), None

    with col_chat:
        render_chat_area()

    if col_evidence is not None:
        with col_evidence:
            render_retrieved_evidence()

    question = st.chat_input("Ask anything about your document...")
    if question:
        if not st.session_state.doc_id:
            st.warning("Please upload a PDF before asking a question.")
        else:
            ask_question(question)
            st.rerun()


if __name__ == "__main__":
    # Streamlit executes the script as __main__, so this is equivalent to
    # calling main() unconditionally when run via `streamlit run app.py` -
    # but it also means `import app` (e.g. from a test) does not trigger the
    # whole UI to render.
    main()