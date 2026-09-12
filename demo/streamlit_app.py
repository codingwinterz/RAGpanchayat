"""Streamlit Chat Interface for the Constitution of India RAG Chatbot."""

from typing import Any
import streamlit as st
import requests

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


def check_backend_health(health_url: str) -> tuple[bool, str]:
    """Perform a health check on the backend API endpoint.

    Args:
        health_url: URL to the /health endpoint.

    Returns:
        A tuple of (is_healthy, status_message).
    """
    try:
        resp: requests.Response = requests.get(health_url, timeout=2)
        if resp.status_code == 200:
            return True, "Backend Connected"
        return False, f"Backend status: {resp.status_code}"
    except Exception:
        return False, "Backend Offline. Start via:\n`uvicorn app.main:app --reload`"


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url: str = st.text_input("Backend API Endpoint", value="http://127.0.0.1:8000/ask")
    health_url: str = api_url.replace("/ask", "/health")

    # Quick health check
    is_healthy, status_msg = check_backend_health(health_url)
    if is_healthy:
        st.success(f"✅ {status_msg}", icon="🟢")
    elif "status:" in status_msg:
        st.warning(f"⚠️ {status_msg}")
    else:
        st.error(f"❌ {status_msg}")

    st.divider()
    st.subheader("💡 Example Queries")
    examples: list[str] = [
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
prompt: str | None = st.chat_input("Ask a question about the Constitution of India...")
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
                response: requests.Response = requests.post(
                    api_url,
                    json={"question": prompt},
                    timeout=30,
                )
                if response.status_code == 200:
                    data: dict[str, Any] = response.json()
                    answer: str = data.get("answer", "No answer received.")
                    cited: list[str] = data.get("cited_articles", [])
                    sources: list[dict[str, Any]] = data.get("retrieved_sources", [])

                    st.markdown(answer)

                    if cited:
                        st.markdown(
                            "**Cited Articles:** "
                            + " ".join([f"`Article {c}`" for c in cited])
                        )

                    if sources:
                        with st.expander("🔍 Retrieved Source Excerpts"):
                            for s in sources:
                                st.markdown(
                                    f"- **Article {s.get('article_number')}:** {s.get('title')} "
                                    f"*(distance: {s.get('distance', 'N/A')})*"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": cited,
                        "sources": sources,
                    })
                else:
                    error_msg: str = f"API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
            except requests.exceptions.ConnectionError:
                conn_err_msg: str = (
                    "Could not connect to the FastAPI server at "
                    f"`{api_url}`. Please make sure it is running via:\n\n"
                    "```bash\nuvicorn app.main:app --reload\n```"
                )
                st.error(conn_err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": conn_err_msg,
                })
            except Exception as e:
                gen_err_msg: str = f"An unexpected error occurred: {str(e)}"
                st.error(gen_err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": gen_err_msg,
                })
