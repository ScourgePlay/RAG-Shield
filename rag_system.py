from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA


load_dotenv()

def build_rag_system():
    print("=" * 50)
    print("Building RAG System... ")
    print("=" * 50)

# Step 1: Load document
    print("\n[1/4] Loading document...")
    loader = TextLoader("documents/company_policy.txt")
    documents = loader.load()
    print(f"      Loaded {len(documents)} document(s)")

# Step 2: Split into chunks
    print("\n[2/4] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks")

# Step 3: Store in vector database
    print("\n[3/4] Building vector database...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("      Vector database created successfully")

# Step 4: Build QA chain
    print("\n[4/4] Connecting LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    print("      LLM connected successfully")