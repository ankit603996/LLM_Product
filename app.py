import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import requests
import os
import PyPDF2
from bs4 import BeautifulSoup
from langchain.schema import Document
from dotenv import load_dotenv
import tempfile
from openai import OpenAIError

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enterprise Chatbot",
    page_icon="🤖",
    layout="wide"
)


# Initialize session state
def init_session_state():
    session_state_vars = {
        "messages": [],
        "vectorstore": None,
        "chat_history": [],
        "generated": [],
        "past": []
    }

    for key, value in session_state_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Call initialization function
init_session_state()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Please set your OpenAI API key in the .env file")
    st.stop()

# Initialize OpenAI components
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )
except Exception as e:
    st.error(f"Error initializing OpenAI components: {str(e)}")
    st.stop()

# RAG prompt template
prompt_template = """Use the following pieces of context and the chat history to answer the question. If the context doesn't contain the answer, use your general knowledge but indicate this clearly.

Previous conversation:
{chat_history}

Context:
{context}

Question: {input}
Answer: """

PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "input"],
    template=prompt_template
)


def extract_text_from_pdf(uploaded_file):
    documents = []
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Process the PDF
        try:
            with open(tmp_file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "page_number": i + 1,
                                "source": uploaded_file.name
                            }
                        ))
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                st.warning(f"Warning: Could not delete temporary file: {str(e)}")

    except Exception as e:
        st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        return []

    return documents


def crawl_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from paragraphs and other relevant tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article'])
        content = "\n".join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])

        if not content:
            st.warning(f"No readable content found at {url}")
            return []

        return [Document(page_content=content, metadata={"source": url})]
    except requests.RequestException as e:
        st.error(f"Failed to fetch website content: {e}")
        return []


def process_documents(documents):
    if not documents:
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_documents = text_splitter.split_documents(documents)

        if not split_documents:
            st.warning("No content to process after splitting documents")
            return None

        return FAISS.from_documents(split_documents, embeddings)
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None


def get_answer(question):
    try:
        if st.session_state.vectorstore:
            # Get chat history string
            chat_history = "\n".join([
                f"Human: {q}\nAssistant: {a}"
                for q, a in zip(st.session_state.past, st.session_state.generated)
            ])

            # RAG-based response
            docs = st.session_state.vectorstore.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            # Format prompt with chat history
            formatted_prompt = PROMPT.format(
                chat_history=chat_history,
                context=context,
                input=question
            )

            return llm.invoke(formatted_prompt).content
        else:
            # Direct LLM response when no context is available
            return llm.invoke(question).content
    except OpenAIError as e:
        return f"OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"Error generating response: {str(e)}"



# Streamlit UI
st.title("🤖 Enterprise Chatbot")

# Sidebar for file upload and URL input
with st.sidebar:
    st.header("📚 Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to create the knowledge base"
    )

    url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com",
        help="Enter a website URL to extract content"
    )

    if st.button("🔄 Process Inputs", help="Click to process uploaded files and URL"):
        with st.spinner("Processing inputs..."):
            documents = []

            # Process PDFs
            for uploaded_file in uploaded_files:
                docs = extract_text_from_pdf(uploaded_file)
                documents.extend(docs)
                if docs:
                    st.sidebar.success(f"Processed {uploaded_file.name}")

            # Process URL
            if url:
                docs = crawl_website(url)
                documents.extend(docs)
                if docs:
                    st.sidebar.success(f"Processed {url}")

            if documents:
                vectorstore = process_documents(documents)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.sidebar.success("✅ Knowledge base updated successfully!")
            else:
                st.sidebar.error("❌ No valid inputs provided")

    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.generated = []
        st.session_state.past = []
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared!")

# Chat input
question = st.chat_input("Ask your question here")

# Initialize containers for chat history and new responses
chat_container = st.container()

# Display chat history
with chat_container:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(st.session_state['generated'][i], key=str(i))

# Handle new question
if question:
    # Display user message immediately
    with chat_container:
        message(question, is_user=True, key=f"{len(st.session_state['past'])}_user")

    # Generate response
    with st.spinner("Thinking..."):
        response = get_answer(question)

    # Display assistant response immediately
    with chat_container:
        message(response, key=str(len(st.session_state['generated'])))

    # Update session state after displaying messages
    st.session_state.past.append(question)
    st.session_state.generated.append(response)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)