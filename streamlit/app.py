from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Streamlit app title
st.title("PDF Q&A with LangChain")

# Upload file section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing the uploaded file..."):
        # Save the uploaded file temporarily
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader("uploaded_pdf.pdf")
        docs = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        # Use Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Store embeddings in FAISS
        vector = FAISS.from_documents(documents, embeddings)

        # Use Hugging Face LLM
        model_name = "EleutherAI/gpt-neo-125M"  # Replace with your desired model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create the pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.01)
        llm = HuggingFacePipeline(pipeline=pipe)

        # Define a prompt template
        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template="""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}"""
        )

        # Create a retrieval-based chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.success("File processed successfully!")

    # Query input
    question = st.text_input("Ask a question based on the uploaded PDF:")

    if question:
        with st.spinner("Generating response..."):
            response = retrieval_chain.invoke({"input": question})
            answer = response["answer"]

        st.write("### Response:")
        st.write(answer)

# Note for users
st.info("Upload a PDF and ask questions based on its content.")
