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
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Load models and setup pipelines
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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

# Global variable for retriever
retriever = None


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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)

    vector = FAISS.from_documents(documents, embeddings)
    global retriever
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    return jsonify({'message': 'File uploaded and processed successfully'}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    if not retriever:
        return jsonify({'error': 'Please upload a PDF first'}), 400

    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return jsonify({'answer': response["answer"]})


@app.route('/ask_direct', methods=['POST'])
def ask_direct_question():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    response = llm.invoke(question)
    return jsonify({'answer': response})


if __name__ == '__main__':
    app.run(debug=True)