import os
import streamlit as st
import requests
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #1A3A5C;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 15px;
        line-height: 1.7;
        color: #1a1a1a;
    }
    .source-box {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #444;
    }
    .source-label {
        font-size: 11px;
        font-weight: 600;
        color: #1A3A5C;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .stat-chip {
        display: inline-block;
        background: #e8f0fe;
        color: #1A3A5C;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 6px;
    }
    .empty-state {
        text-align: center;
        color: #888;
        padding: 40px 20px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the backend is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint, **kwargs):
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the backend is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The LLM is still loading — try again in 30 seconds.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_delete(endpoint):
    try:
        r = requests.delete(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Documents")
    st.markdown("---")

    # Upload section
    st.markdown("**Upload a document**")
    uploaded = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("➕ Index this document", use_container_width=True, type="primary"):
            with st.spinner(f"Indexing {uploaded.name}..."):
                result = api_post(
                    "/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
            if result:
                st.success(f"✅ Indexed {result['chunks_added']} chunks")
                st.rerun()

    st.markdown("---")

    # Indexed documents list
    st.markdown("**Indexed documents**")
    sources_data = api_get("/sources")

    if sources_data and sources_data["total"] > 0:
        st.markdown(
            f'<span class="stat-chip">{sources_data["total"]} document{"s" if sources_data["total"] != 1 else ""}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        for src in sources_data["sources"]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"📄 `{src}`")
            with col2:
                if st.button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                    result = api_delete(f"/source/{src}")
                    if result:
                        st.success(f"Removed {src}")
                        st.rerun()
    else:
        st.markdown(
            '<div class="empty-state">No documents indexed yet.<br>Upload one above to get started.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # API status
    health = api_get("/health")
    if health and health.get("status") == "healthy":
        st.markdown("🟢 **API online**")
    else:
        st.markdown("🔴 **API offline**")


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Document Intelligence")
st.markdown("Ask questions about your uploaded documents. Answers are grounded in your content — the AI cannot make things up.")
st.markdown("---")

if not sources_data or sources_data["total"] == 0:
    st.info("👈 Upload a document in the sidebar to get started.")
else:
    # Question input
    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the main technical skills? What is the project about? Summarise the key findings.",
        height=100,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider(
            "Number of source chunks to retrieve",
            min_value=1,
            max_value=6,
            value=3,
            help="More chunks = more context for the LLM but slower response",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        ask = st.button("Ask →", type="primary", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching documents and generating answer..."):
                start = time.time()
                result = api_post(
                    "/query",
                    json={"question": question, "top_k": top_k},
                )
                elapsed = round(time.time() - start, 1)

            if result:
                # Answer
                st.markdown("### Answer")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<span class="stat-chip">⏱ {elapsed}s</span>', unsafe_allow_html=True)

                # Sources
                st.markdown("### Sources used")
                st.markdown(f"*Retrieved {len(result['sources'])} chunk(s) from your documents:*")

                for i, src in enumerate(result["sources"], 1):
                    with st.expander(f"Chunk {i} — {src['source']}"):
                        st.markdown(
                            f'<div class="source-box"><div class="source-label">📄 {src["source"]}</div>{src["content"]}</div>',
                            unsafe_allow_html=True,
                        )

    # Chat history tip
    st.markdown("---")
    st.markdown(
        "*Tip: The more specific your question, the better the answer. Try asking about specific sections, people, or topics mentioned in your document.*"
    )
