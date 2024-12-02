# https://vijaykumarkartha.medium.com/beginners-guide-to-retrieval-chain-from-langchain-f307b1a20e77
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

pdf_path = "COL- Statistical Methods of Decision Making - Prof. P K Vishwanathan.pdf"  # Replace with your PDF file path
loader = PyPDFLoader(pdf_path)
docs = loader.load()
# Create a text splitter with a fixed chunk size
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import FAISS
vector = FAISS.from_documents(documents, embeddings)
# Use Hugging Face LLM
model_name = "EleutherAI/gpt-neo-125M"  # Replace with "EleutherAI/gpt-neo-125M" or similar if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use pipeline for inference
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.01)
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}"""
)

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm,prompt)
retriever = vector.as_retriever()
from langchain.chains import create_retrieval_chain
import sys
import io

# Ensure UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input":"What are the talent products delivered by DASA"})
print (response["answer"].encode('utf-8').decode('utf-8'))

### Without context
# Generate a response using the LLM
response = llm("What are the talent products delivered by DASA")
print(response)
