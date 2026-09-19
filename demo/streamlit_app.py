"""Streamlit Chat Interface for the Constitution of India RAG Chatbot."""

import os
# pyrefly: ignore [missing-import]
import streamlit as st
import requests

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/ask")

st.set_page_config(
    page_title="Constitution of India — RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Constitution of India — Legal RAG Assistant")
st.caption(
    "Ask legal queries strictly grounded in the Constitution of India. "
    "Every answer includes verified article citations."
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("Backend API Endpoint", value=DEFAULT_BACKEND_URL)
    health_url = api_url.replace("/ask", "/health")

    # Quick health check
    try:
        resp = requests.get(health_url, timeout=2)
        if resp.status_code == 200:
            st.success("✅ Backend Connected", icon="🟢")
        else:
            st.warning(f"⚠️ Backend status: {resp.status_code}")
    except Exception:
        st.error("❌ Backend Offline. Start via:\n`uvicorn app.main:app --reload`")

    st.divider()
    st.subheader("💡 Example Queries")
    examples = [
        "What protections are provided under Article 21?",
        "Can the President grant pardons and remit sentences?",
        "What are the Fundamental Duties of Indian citizens?",
        "What constitutes a Money Bill in Parliament?",
        "What does the Constitution say about a Uniform Civil Code?",
        "How can Parliament amend the Constitution?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["pending_prompt"] = ex

# Chat message state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! I am your domain-locked assistant for the Constitution of India. Ask any question, and I will retrieve and cite the exact Articles.",
            "citations": [],
            "sources": [],
        }
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.markdown(
                "**Cited Articles:** "
                + " ".join([f"`Article {c}`" for c in msg["citations"]])
            )
        if msg.get("sources"):
            with st.expander("🔍 Retrieved Source Excerpts"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **Article {s.get('article_number')}:** {s.get('title')} "
                        f"*(distance: {s.get('distance', 'N/A')})*"
                    )

# Determine user input (from chat box or example button)
prompt = st.chat_input("Ask a question about the Constitution of India...")
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching Constitution articles and generating answer..."):
            try:
                response = requests.post(
                    api_url,
                    json={"question": prompt},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    cited = data.get("cited_articles", [])
                    verified = data.get("verified_articles", [])
                    unverified = data.get("unverified_articles", [])
                    sources = data.get("retrieved_sources", [])

                    st.markdown(answer)

                    if cited:
                        unverified_set = set(unverified)
                        badges = " ".join(
                            f"`Article {c}` ✅" if (c in verified or c not in unverified_set)
                            else f"`Article {c}` ⚠️"
                            for c in cited
                        )
                        st.markdown(f"**Cited Articles:** {badges}")

                    if unverified:
                        st.warning(
                            "⚠️ Some cited articles ("
                            + ", ".join(f"Article {c}" for c in unverified)
                            + ") were not in the retrieved context — treat them with caution."
                            )

                    if sources:
                        with st.expander("🔍 Retrieved Source Excerpts"):
                            for s in sources:
                                if s.get("retrieval") == "keyword_only" or s.get("distance") is None:
                                    dist_text = "BM25 keyword-only hit"
                                else:
                                    dist_text = f"distance: {s.get('distance')}"
                                st.markdown(
                                    f"- **Article {s.get('article_number')}:** {s.get('title')} "
                                    f"*({dist_text})*"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": cited,
                        "sources": sources,
                    })
                else:
                    error_msg = f"API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
            except requests.exceptions.ConnectionError:
                error_msg = (
                    "Could not connect to the FastAPI server at "
                    f"`{api_url}`. Please make sure it is running via:\n\n"
                    "```bash\nuvicorn app.main:app --reload\n```"
                )
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
