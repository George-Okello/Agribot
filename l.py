import os
from dotenv import load_dotenv
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationEntityMemory
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.cohere import Cohere
import chainlit as cl
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
# Import FAISSManager from the separate file
from faiss_manager import FAISSManager
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize the LLM
# llm = ChatOllama(
#     model="gemma:2b",
#     temperature=0,
# )
llm = Cohere(temperature=0, cohere_api_key='4uMu3uV6GDBPnNvVVrqJiSgrWuCWWkJPjQix5NiJ')
# Initialize FAISS and embeddings
faiss_manager = FAISSManager(
    excel_path="Data/Weather Agro Advisory Knowledge Base(1).xlsx",
    faiss_index_path="faiss_index"
)

# Check if the FAISS index already exists; if not, initialize it
if not os.path.exists("faiss_index"):
    faiss_manager.initialize_faiss()

# Load FAISS index
faiss_db = faiss_manager.load_faiss_index()

# Load documents if they haven't been loaded already
if not faiss_manager.get_docs():
    # This is to ensure documents are available even if loaded from an existing FAISS index
    faiss_manager.initialize_faiss()

# Now, get the loaded documents
docs = faiss_manager.get_docs()

# Define the retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=faiss_db.as_retriever(), llm=llm
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:""")
])


# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Create the Retrieval-Augmented Generation (RAG) chain
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Example usage
retrieved_docs = rag_chain.invoke("when to plant maize?")
print(retrieved_docs)
