"""Retrieval over Northbridge's published policy wordings, stored in Qdrant."""
import os

from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://qdrant:6333"))

store = Qdrant(client=qdrant, collection_name="policy-wordings", embeddings=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 4})


def retrieve_context(question: str, policy_type: str) -> str:
    docs = retriever.invoke(question, filter={"policy_type": policy_type})
    return "\n\n".join(d.page_content for d in docs)


def index_wordings(paths: list[str]) -> None:
    """Run by the publishing team whenever a policy wording changes."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for path in paths:
        chunks = splitter.split_documents(PyPDFLoader(path).load())
        store.add_documents(chunks)
