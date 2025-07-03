import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import textwrap
from langchain.prompts import PromptTemplate
import streamlit.components.v1 as components
import uuid
import time

# Load environment
load_dotenv()

# Set Streamlit UI
st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")

# Define prompt
prompt_template = """
You are a senior Indian tax and legal assistant. You provide structured, legally accurate, and clear answers to complex questions.

Use your **tax/legal knowledge** to do the following:

1. Clearly state the **applicable law** or sections (e.g., Income Tax Act, company law, etc.)
2. **Apply the law** logically to the user's scenario step-by-step
3. If available, **cite relevant case law or judicial decisions** from the context or your knowledge
4. Mention **exceptions** or interpretations if applicable
5. Use **structured formatting** (bullets, headers) for clarity
6. Provide a **confident, reasoned conclusion** — avoid vague or advisory statements like “consult a tax expert”

⚖️ If multiple viewpoints or legal interpretations exist, mention each and clarify which one applies best.

Do not quote the document blindly. Extract principles and use them to reason.

---

Context:
{context}

Question:
{question}

Structured Legal Answer:
"""
custom_prompt = PromptTemplate.from_template(prompt_template)

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local("faiss-vdbs/combined", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

vectordb = load_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# --- Chat session manager ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if st.session_state.current_session_id not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.current_session_id] = {
        "title": "New Chat",
        "messages": []
    }

def get_session_title(messages):
    for m in messages:
        if m["role"] == "user":
            return m["content"][:30] + "..." if len(m["content"]) > 30 else m["content"]
    return "Chat " + time.strftime("%H:%M:%S")

# --- Sidebar ---
st.sidebar.title("💼 Legal Chat Sessions")
if st.sidebar.button("➕ New Chat"):
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.chat_sessions[st.session_state.current_session_id] = {
        "title": "New Chat",
        "messages": []
    }

delete_id = None
for session_id, session in st.session_state.chat_sessions.items():
    if session_id == st.session_state.current_session_id:
        st.sidebar.markdown(f"**▶ {session['title']}**")
    else:
        cols = st.sidebar.columns([0.8, 0.2])
        if cols[0].button(session["title"], key=session_id):
            st.session_state.current_session_id = session_id
        if cols[1].button("❌", key=session_id + "_delete"):
            delete_id = session_id

if delete_id:
    del st.session_state.chat_sessions[delete_id]
    if delete_id == st.session_state.current_session_id:
        if st.session_state.chat_sessions:
            st.session_state.current_session_id = next(iter(st.session_state.chat_sessions))
        else:
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_sessions[st.session_state.current_session_id] = {
                "title": "New Chat",
                "messages": []
            }

# --- Main Chat Interface ---
st.title("📜 Legal & Tax Chatbot")
session = st.session_state.chat_sessions[st.session_state.current_session_id]
user_input = st.chat_input("Ask a legal/tax question...")

# Display history
for msg in session["messages"]:
    if msg["role"] in ("user", "assistant"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    elif msg["role"] == "sources":
        st.markdown("### 📄 Sources")
        for i, doc in enumerate(msg["content"], start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("pageNum", "?")
            passage = doc.page_content.strip()
            wrapped_passage = textwrap.fill(passage, width=100)
            with st.expander(f"📄 Source {i}: {source} — Page {page}"):
                st.markdown(wrapped_passage)

# Scroll to bottom
components.html(
    """<script>
        var element = window.parent.document.querySelector('.main');
        element.scrollTo({ top: element.scrollHeight, behavior: 'smooth' });
    </script>""",
    height=0,
)

# Process new input
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    session["messages"].append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = qa_chain({"query": user_input})
        bot_answer = response["result"]
        sources = response.get("source_documents", [])

    with st.chat_message("assistant"):
        st.markdown(bot_answer)
    session["messages"].append({"role": "assistant", "content": bot_answer})

    if sources:
        st.markdown("### 📄 Sources")
        for i, doc in enumerate(sources, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("pageNum", "?")
            passage = doc.page_content.strip()
            wrapped_passage = textwrap.fill(passage, width=100)
            with st.expander(f"📄 Source {i}: {source} — Page {page}"):
                st.markdown(wrapped_passage)
        session["messages"].append({"role": "sources", "content": sources})

    # Auto-title after first user message
    if session["title"] == "New Chat":
        session["title"] = get_session_title(session["messages"])