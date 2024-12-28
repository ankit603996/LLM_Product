from flask import Flask, request, render_template, jsonify, redirect, url_for
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import requests
import os
import PyPDF2
from bs4 import BeautifulSoup
from langchain.schema import Document
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            template_folder=os.path.abspath('templates'),
            static_folder=os.path.abspath('static'))

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not OPENAI_API_KEY:
    logger.error("Missing OpenAI API Key")
    raise ValueError("Missing OpenAI API Key")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or "gpt-4" if you have access
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# Prompt template for RAG
prompt_template = """Answer the following question based only on the provided context:
BEGIN CONTEXT
{context}
END CONTEXT
Question: {input}
Answer:"""


@app.route('/')
def home():
    try:
        logger.debug("Rendering home template")
        return render_template('home.html')
    except Exception as e:
        logger.error(f"Error rendering home template: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/qa')
def qa():
    try:
        logger.debug("Rendering qa template")
        return render_template('qa.html')
    except Exception as e:
        logger.error(f"Error rendering qa template: {str(e)}")
        return jsonify({'error': str(e)}), 500


def extract_text_from_pdf(file_path):
    documents = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            documents.append(Document(page_content=text, metadata={"page_number": i + 1}))
    return documents


def crawl_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text() for p in paragraphs])
        return [Document(page_content=content, metadata={"source": url})]
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch website content: {e}")


@app.route('/upload', methods=['POST'])
def upload_inputs():
    files = request.files.getlist('files')
    url = request.form.get('url')

    logger.debug(f"Received upload request - Files: {[f.filename for f in files]}, URL: {url}")

    documents = []

    # Process each uploaded PDF
    for file in files:
        if file and file.filename != '':
            try:
                import tempfile
                file_path = os.path.join(tempfile.gettempdir(), file.filename)
                file.save(file_path)
                logger.debug(f"Processing PDF file: {file_path}")
                documents += extract_text_from_pdf(file_path)
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error processing PDF {file.filename}: {str(e)}")
                return jsonify({'error': f"Error processing PDF {file.filename}: {str(e)}"}), 500

    # Process URL if provided
    if url and url.strip():
        try:
            logger.debug(f"Processing URL: {url}")
            documents += crawl_website(url)
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return jsonify({'error': f"Error processing URL: {str(e)}"}), 500

    if not documents:
        return jsonify({'error': 'No valid inputs provided'}), 400

    try:
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(documents)

        # Create vector store
        vector = FAISS.from_documents(split_documents, embeddings)
        global retriever
        retriever = vector.as_retriever(search_kwargs={"k": 10})

        logger.info("Successfully processed all inputs")
        return jsonify({'message': 'Inputs processed successfully'}), 200

    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        return jsonify({'error': f"Failed to process inputs: {str(e)}"}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Retrieve relevant context
        docs = retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in docs])

        # Format prompt with context and question
        formatted_prompt = prompt_template.format(context=context, input=question)

        # Get response from OpenAI
        response = llm.predict(formatted_prompt)
        return jsonify({'answer': response})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f"Error processing request: {str(e)}"}), 500


@app.route('/ask_direct', methods=['POST'])
def ask_direct_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Get direct response from OpenAI
        response = llm.predict(question)
        return jsonify({'answer': response})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f"Error processing request: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)