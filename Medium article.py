## https://vijaykumarkartha.medium.com/beginners-guide-to-retrieval-chain-from-langchain-f307b1a20e77
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.dasa.org")
docs = loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
from langchain_community.vectorstores import FAISS
vector = FAISS.from_documents(documents, embeddings)

#Creating a Retrieval Chain
# Import ChatOpenAI and create an llm with the Open AI API key
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key="<Your Open AI API Key here>")
# Create a prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm,prompt)
retriever = vector.as_retriever()
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input":"What are the talent products delivered by DASA"})
print (response["answer"])
