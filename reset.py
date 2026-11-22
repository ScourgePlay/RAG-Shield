"""
reset.py: rebuilds the clean corpus from the documents folder.

It deletes the vector store, splits the policy documents into passages,
stores them again, and saves the approved list used by the integrity check.

It does not load the trained models, so it is also the first command to run
on a fresh copy of the project.

Usage:
    python reset.py
"""

import os
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from defense2b import get_embeddings
from integrity import save_baseline

# These four settings must stay the same as in rag_system.py.
CHROMA_PATH = "./chroma_db"
DOCS_DIR = "documents"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def main():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    loader = DirectoryLoader(
        DOCS_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(loader.load())

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
    saved = save_baseline([c.page_content for c in chunks])
    print(f"Corpus rebuilt: {store._collection.count()} passages, "
          f"{saved} saved to the approved list.")


if __name__ == "__main__":
    main()
