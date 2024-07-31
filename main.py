import os
from dotenv import load_dotenv
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOllama

# Import FAISSManager from the separate file
from faiss_manager import FAISSManager

# Load environment variables
load_dotenv()

# Initialize FAISS and embeddings
faiss_manager = FAISSManager(
    excel_path="Data/Weather Agro Advisory Knowledge Base(1).xlsx",
    faiss_index_path="faiss_index"
)

# Check if the FAISS index already exists, if not initialize it
if not os.path.exists("faiss_index"):
    faiss_manager.initialize_faiss()

# Load the FAISS index
faiss_db = faiss_manager.load_faiss_index()
llm = ChatOllama(
    model="llama3",
    temperature=0,
    # other params...
)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_db.as_retriever())
question = "Tell me about maize"
result = chain({"question": question})

print(result['answer'])
print("\n Sources : ", result['sources'])
