import os

import chainlit as cl
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.tracers.log_stream import LogStreamCallbackHandler

# Import FAISSManager from the separate file
from faiss_manager import FAISSManager

# Load environment variables
load_dotenv()

# LLM configuration
llm = ChatOllama(
    model="gemma:2b",
    temperature=0,
)

# FAISS Manager setup
faiss_manager = FAISSManager(
    excel_path="Data/Weather Agro Advisory Knowledge Base(1).xlsx",
    faiss_index_path="faiss_index"
)

# Check if the FAISS index already exists; if not, initialize it
if not os.path.exists("faiss_index"):
    faiss_manager.initialize_faiss()

# Load the FAISS index and set up retriever
faiss_db = faiss_manager.load_faiss_index()
retriever = faiss_db.as_retriever(search_kwargs={"k": 4})

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

# Run
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
chain.invoke(docs)