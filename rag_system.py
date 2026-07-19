from dotenv import load_dotenv
import logging
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_rag_system():
    logger.info("=" * 50)
    logger.info("Building RAG System...")
    logger.info("=" * 50)

    # Step 1: Load document
    logger.info("[1/4] Loading document...")
    loader = TextLoader("documents/company_policy.txt")
    documents = loader.load()
    logger.info("Loaded %d document(s)", len(documents))

    # Step 2: Split into chunks
    logger.info("[2/4] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    # Step 3: Store in vector database
    logger.info("[3/4] Building vector database...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    logger.info("Vector database created successfully")

    # Step 4: Build QA chain
    logger.info("[4/4] Connecting LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    logger.info("LLM connected successfully")

    return qa_chain

def ask_question(qa_chain, question):
    """Send a question to the RAG system and print the answer + sources."""
    result = qa_chain.invoke({"query": question})
    print("\nAnswer:", result["result"])
    print("\nSources used:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"  [{i}] {doc.page_content[:80].strip()}...")
    return result


if __name__ == "__main__":
    try:
        qa_chain = build_rag_system()
    except Exception:
        logger.exception("RAG system build failed")
        raise

    print("\n--- RAG System Ready. Type a question, or 'quit' to exit. ---")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not question:
            continue
        ask_question(qa_chain, question)