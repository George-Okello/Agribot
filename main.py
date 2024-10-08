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

# Load the FAISS index
# faiss_db = faiss_manager.load_faiss_index()
#
# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_db.as_retriever())
# question = "maize"
# result = chain({"question": question})
#
# print(result['answer'])
# print("\n Sources : ", result['sources'])
llm = ChatOllama(
    model="gemma:2b",
    temperature=0,
)
# Initialize FAISS and embeddings
faiss_manager = FAISSManager(
    excel_path="Data/Weather Agro Advisory Knowledge Base(1).xlsx",
    faiss_index_path="faiss_index"
)
# Check if the FAISS index already exists, if not initialize it
if not os.path.exists("faiss_index"):
    faiss_manager.initialize_faiss()
faiss_db = faiss_manager.load_faiss_index()
retriever = MultiQueryRetriever.from_llm(
    retriever=faiss_db.as_retriever(), llm=llm
)
# retriever = faiss_db.as_retriever(search_kwargs={"k": 4})
# Retrieve and generate using the relevant snippets of the blog.
prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:""")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
retrieved_docs=rag_chain.invoke("when to plant maize?")
print(retrieved_docs)
# @cl.on_chat_start
# async def init():
#     retriever = faiss_db.as_retriever(search_kwargs={"k": 4})
#     compressor = CohereRerank(user_agent="my-app", cohere_api_key='4uMu3uV6GDBPnNvVVrqJiSgrWuCWWkJPjQix5NiJ')
#     reranker = ContextualCompressionRetriever(
#         base_compressor=compressor, base_retriever=retriever, document=faiss_db)
#
#     memory = ConversationEntityMemory(
#         llm=Cohere(temperature=0, cohere_api_key='4uMu3uV6GDBPnNvVVrqJiSgrWuCWWkJPjQix5NiJ'),
#         return_messages=True
#     )
#
#     runnable = RetrievalQA.from_chain_type(
#         llm=Cohere(temperature=0, cohere_api_key='4uMu3uV6GDBPnNvVVrqJiSgrWuCWWkJPjQix5NiJ'), retriever=reranker,
#         chain_type="refine", memory=memory,
#     )
#
#     cl.user_session.set("query", runnable)
#
#
# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("query")  # type: Runnable
#     cb = cl.AsyncLangchainCallbackHandler()
#     res = await runnable.acall(message.content, callbacks=[cb])
#
#     await cl.Message(content=res["result"]).send()

#%%

#%%
