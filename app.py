from flask import Flask, request, render_template, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import requests
import os
import PyPDF2
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
app = Flask(__name__)
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI Configuration
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
ENDPOINT_EMBEDDING = os.getenv("ENDPOINT_EMBEDDING")
API_KEY_EMBEDDING = os.getenv("API_KEY_EMBEDDING")
HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Custom Embedding Class for Azure OpenAI
class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

    def embed_documents(self, texts):
        payload = {
            "input": texts
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            embeddings = response.json()["data"]
            return [item["embedding"] for item in embeddings]
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch embeddings from Azure OpenAI: {e}")

    def embed_query(self, query):
        return self.embed_documents([query])[0]  # Assuming you want to embed a single query


# Instantiate the custom embeddings class
embeddings = AzureOpenAIEmbeddings(API_KEY_EMBEDDING, ENDPOINT_EMBEDDING)

# Prompt template
prompt_template = """Answer the following question based only on the provided context:
BEGIN CONTEXT
{context}
END CONTEXT
Question: {input}
Answer:"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    import tempfile

    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(file_path)

    # Process the PDF
    documents = []
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()  # Extract text from the page
                documents.append(Document(page_content=text, metadata={"page_number": i + 1}))
    except Exception as e:
        return jsonify({'error': f"Failed to process PDF: {str(e)}"}), 500

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(documents)

    # Create a vector store using custom Azure OpenAI embeddings
    vector = FAISS.from_documents(split_documents, embeddings)  # Use AzureOpenAIEmbeddings with FAISS
    global retriever
    retriever = vector.as_retriever(search_kwargs={"k": 10})  # Retrieve top 10 documents

    return jsonify({'message': 'File uploaded and processed successfully'}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Retrieve relevant context
    docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in docs])

    # Prepare payload for Azure OpenAI
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_template.format(context=context, input=question)}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    # Send request to Azure OpenAI
    try:
        response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return jsonify({'answer': answer})
    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch response from Azure OpenAI: {e}"}), 500

@app.route('/ask_direct', methods=['POST'])
def ask_direct_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Prepare payload for Azure OpenAI
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    # Send request to Azure OpenAI
    try:
        response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return jsonify({'answer': answer})
    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch response from Azure OpenAI: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
