
from flask import Flask, request, render_template, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = './uploads'
from dotenv import load_dotenv
load_dotenv()
print(f"Token: {os.getenv('HUGGINGFACEHUB_API_TOKEN')}")
# Load models and setup pipelines globally

# Load models and setup pipelines using Hugging Face API
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Replace local model with Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"max_new_tokens": 100, "temperature": 0.01},
)
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""Answer the following question based only on the provided context:
BEGIN CONTEXT
{context}
END CONTEXT
Question: {input}"""
)


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
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    global retriever
    retriever = vector.as_retriever()


    return jsonify({'message': 'File uploaded and processed successfully'}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    # Store retriever globally (or use session storage)
    document_chain = create_stuff_documents_chain(llm, prompt)
    print(question)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return jsonify({'answer': response["answer"]})


@app.route('/ask_direct', methods=['POST'])
def ask_direct_question():
    """
    Directly uses LLM to generate answers without retrieving context from the uploaded PDF.
    """
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Direct LLM invocation without retrieval
    response = llm.invoke(question)
    return jsonify({'answer': response})


if __name__ == '__main__':
#    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
