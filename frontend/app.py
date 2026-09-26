import re

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📄",
    layout="wide",
)


st.title("📄 Multimodal RAG")
st.caption(
    "Ask questions about your PDF using text, tables, and images."
)


# ---------------------------------------------------------
# Helper: validate citation IDs used by the model
# ---------------------------------------------------------

def extract_citation_ids(answer: str) -> list[str]:
    """
    Extract citations such as [S1], [S2], [I1], [I2]
    from the generated answer.
    """
    return re.findall(
        r"\[(?:S|I)\d+\]",
        answer,
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if st.button(
            "📥 Ingest PDF",
            use_container_width=True,
        ):
            with st.spinner("Processing PDF..."):
                try:
                    response = requests.post(
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

                    if response.status_code == 200:
                        result = response.json()

                        st.session_state["doc_id"] = (
                            result["doc_id"]
                        )

                        st.session_state["document_name"] = (
                            result["source_filename"]
                        )

                        st.success(
                            "PDF ingested successfully!"
                        )

                        st.write(
                            f"**Pages:** {result['num_pages']}"
                        )

                        st.write(
                            f"**Text chunks:** "
                            f"{result['text_chunks']}"
                        )

                        st.write(
                            f"**Tables:** {result['tables']}"
                        )

                        st.write(
                            f"**Images:** {result['images']}"
                        )

                    else:
                        st.error(
                            f"Ingestion failed: "
                            f"{response.text}"
                        )

                except requests.exceptions.RequestException as e:
                    st.error(
                        f"Could not connect to FastAPI: {e}"
                    )

    if "doc_id" in st.session_state:
        st.divider()

        st.write("### Current document")

        st.write(
            f"📄 {st.session_state['document_name']}"
        )

        st.caption(
            f"Document ID: "
            f"{st.session_state['doc_id']}"
        )


# ---------------------------------------------------------
# Main question interface
# ---------------------------------------------------------

if "doc_id" not in st.session_state:

    st.info(
        "Upload a PDF from the sidebar and click "
        "**Ingest PDF** to start."
    )

else:

    st.subheader("Ask a question")

    question = st.text_area(
        "Question",
        placeholder=(
            "e.g. What does the proposed model "
            "architecture look like?"
        ),
        height=100,
    )

    ask_button = st.button(
        "🔍 Ask Question",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document and "
                "generating answer..."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "doc_id": st.session_state["doc_id"],
                        },
                        timeout=600,
                    )

                    if response.status_code == 200:

                        result = response.json()

                        st.divider()

                        # -------------------------------------------------
                        # Answer
                        # -------------------------------------------------

                        st.subheader("Answer")

                        answer = result.get(
                            "answer",
                            "No answer returned.",
                        )

                        st.write(answer)

                        # -------------------------------------------------
                        # Citation validation
                        # -------------------------------------------------

                        citations = result.get(
                            "citations",
                            [],
                        )

                        valid_source_ids = {
                            citation.get("source_id")
                            for citation in citations
                            if citation.get("source_id")
                        }

                        answer_citations = (
                            extract_citation_ids(answer)
                        )

                        invalid_citations = [
                            citation
                            for citation in answer_citations
                            if citation.strip("[]")
                            not in valid_source_ids
                        ]

                        if invalid_citations:

                            st.warning(
                                "The model referenced a source ID "
                                "that was not returned by retrieval: "
                                + ", ".join(
                                    sorted(
                                        set(
                                            invalid_citations
                                        )
                                    )
                                )
                            )

                        # -------------------------------------------------
                        # Sources
                        # -------------------------------------------------

                        st.subheader("📚 Sources")

                        if citations:

                            text_sources = [
                                citation
                                for citation in citations
                                if str(
                                    citation.get("source_id", "")
                                ).startswith("S")
                            ]

                            image_sources = [
                                citation
                                for citation in citations
                                if str(
                                    citation.get("source_id", "")
                                ).startswith("I")
                            ]

                            # ---------------------------------------------
                            # Text sources
                            # ---------------------------------------------

                            if text_sources:

                                st.markdown(
                                    "### 📄 Text / Table Sources"
                                )

                                for citation in text_sources:

                                    source_id = citation.get(
                                        "source_id",
                                        "S?",
                                    )

                                    page = citation.get(
                                        "page",
                                        "unknown",
                                    )

                                    source_type = citation.get(
                                        "type",
                                        "text",
                                    )

                                    label = citation.get(
                                        "label",
                                        "Text source",
                                    )

                                    st.markdown(
                                        f"**[{source_id}]** "
                                        f"Page {page} — "
                                        f"{label} "
                                        f"({source_type})"
                                    )

                            # ---------------------------------------------
                            # Image sources
                            # ---------------------------------------------

                            if image_sources:

                                st.markdown(
                                    "### 🖼️ Image Sources"
                                )

                                for citation in image_sources:

                                    source_id = citation.get(
                                        "source_id",
                                        "I?",
                                    )

                                    page = citation.get(
                                        "page",
                                        "unknown",
                                    )

                                    st.markdown(
                                        f"**[{source_id}]** "
                                        f"Page {page} — "
                                        f"Image source"
                                    )

                        else:

                            st.write(
                                "No citations returned."
                            )

                        # -------------------------------------------------
                        # Retrieved text
                        # -------------------------------------------------

                        retrieved_text = result.get(
                            "retrieved_text",
                            [],
                        )

                        if retrieved_text:

                            with st.expander(
                                "🔎 View retrieved text"
                            ):

                                for item in retrieved_text:

                                    source_id = item.get(
                                        "source_id",
                                        "S?",
                                    )

                                    page = item.get(
                                        "page",
                                        "unknown",
                                    )

                                    source_type = item.get(
                                        "type",
                                        "text",
                                    )

                                    st.markdown(
                                        f"**[{source_id}]** "
                                        f"Page {page} — "
                                        f"{source_type}"
                                    )

                                    st.write(
                                        item.get(
                                            "content",
                                            "",
                                        )
                                    )

                                    st.caption(
                                        f"Retrieval distance: "
                                        f"{item.get('distance')}"
                                    )

                                    st.divider()

                        # -------------------------------------------------
                        # Retrieved images
                        # -------------------------------------------------

                        retrieved_images = result.get(
                            "retrieved_images",
                            [],
                        )

                        if retrieved_images:

                            with st.expander(
                                "🖼️ View retrieved images"
                            ):

                                for item in retrieved_images:

                                    source_id = item.get(
                                        "source_id",
                                        "I?",
                                    )

                                    page = item.get(
                                        "page",
                                        "unknown",
                                    )

                                    image_path = item.get(
                                        "file_path"
                                    )

                                    st.markdown(
                                        f"**[{source_id}]** "
                                        f"Page {page} — Image"
                                    )

                                    if image_path:

                                        try:

                                            st.image(
                                                image_path,
                                                caption=(
                                                    item.get(
                                                        "caption"
                                                    )
                                                    or
                                                    f"[{source_id}] "
                                                    f"Page {page}"
                                                ),
                                                use_container_width=True,
                                            )

                                        except Exception:

                                            st.caption(
                                                f"Image path: "
                                                f"{image_path}"
                                            )

                                    st.caption(
                                        f"Retrieval distance: "
                                        f"{item.get('distance')}"
                                    )

                                    st.divider()

                    else:

                        st.error(
                            f"Question failed: "
                            f"{response.text}"
                        )

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Could not connect to FastAPI: {e}"
                    )
