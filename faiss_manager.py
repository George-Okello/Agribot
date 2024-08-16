# import os
# from langchain_community.document_loaders import UnstructuredExcelLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores.faiss import FAISS
#
#
# class FAISSManager:
#     def __init__(self, excel_path: str, faiss_index_path: str,
#                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = 'cpu'):
#         self.excel_path = excel_path
#         self.faiss_index_path = faiss_index_path
#         self.model_name = model_name
#         self.device = device
#         self.db = None
#         self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})
#
#     def initialize_faiss(self):
#         loader = UnstructuredExcelLoader(self.excel_path, mode="elements")
#         docs = loader.load()
#         self.db = FAISS.from_documents(docs, self.embeddings)
#         self.save_faiss_index()
#
#     def save_faiss_index(self):
#         if self.db:
#             self.db.save_local(self.faiss_index_path)
#         else:
#             raise ValueError("FAISS database is not initialized. Call initialize_faiss() first.")
#
#     def load_faiss_index(self):
#         self.db = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
#         return self.db

import os
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS


class FAISSManager:
    def __init__(self, excel_path: str, faiss_index_path: str,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = 'cpu'):
        self.excel_path = excel_path
        self.faiss_index_path = faiss_index_path
        self.model_name = model_name
        self.device = device
        self.db = None
        self.docs = None  # To store the loaded documents
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})

    def initialize_faiss(self):
        # Load documents from the Excel file
        loader = UnstructuredExcelLoader(self.excel_path, mode="elements")
        self.docs = loader.load()

        # Build FAISS index from documents
        self.db = FAISS.from_documents(self.docs, self.embeddings)

        # Save the FAISS index
        self.save_faiss_index()

    def save_faiss_index(self):
        if self.db:
            self.db.save_local(self.faiss_index_path)
        else:
            raise ValueError("FAISS database is not initialized. Call initialize_faiss() first.")

    def load_faiss_index(self):
        # Load the FAISS index from the file
        self.db = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

        # Load documents from the Excel file to ensure self.docs is populated
        if self.docs is None:
            loader = UnstructuredExcelLoader(self.excel_path, mode="elements")
            self.docs = loader.load()

        return self.db

    def get_docs(self):
        if self.docs is None:
            raise ValueError("Documents not loaded. Call initialize_faiss() first.")
        return self.docs
