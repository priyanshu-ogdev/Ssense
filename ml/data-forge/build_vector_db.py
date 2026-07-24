import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_db():
    print("🚀 Initializing Vector Database Builder...")
    
    db_path = "./chroma_db"
    file_path = "./dpdp_act_and_rules_2025.txt"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Legal text file not found at {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("✂️ Chunking legal text using Legal-Aware Separators...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\nSection ", "\n\nRule ", "\n\nCHAPTER", "\n\n", "\n", " "],
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents([text])
    
    print(f"✅ Created {len(chunks)} chunks.")

    print("🧠 Initializing ChromaDB and BAAI/bge-small-en-v1.5 embedding model...")
    # Initialize chroma client
    client = chromadb.PersistentClient(path=db_path)
    
    # We use sentence-transformers BAAI/bge-small-en-v1.5
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    
    # Delete old collection if it exists to keep it fresh
    try:
        client.delete_collection("dpdp_law")
    except Exception:
        pass
        
    collection = client.create_collection(name="dpdp_law", embedding_function=ef)
    
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "dpdp_act_and_rules_2025.txt"} for _ in range(len(chunks))]
    
    print("⏳ Embedding and inserting chunks into ChromaDB (this may take a moment)...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    print("✅ Vector Database built successfully.")
    
    print("\n🔍 Verification Query: 'What is the penalty for data breach?'")
    results = collection.query(
        query_texts=["What is the penalty for data breach?"],
        n_results=2
    )
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n--- Result {i+1} ---\n{doc}")

if __name__ == "__main__":
    build_db()
