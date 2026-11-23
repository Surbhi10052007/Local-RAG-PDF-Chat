import pypdf
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the PDF
print("1. Reading PDF...")
reader = pypdf.PdfReader("test.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# 2. Chunk the text
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
print(f"2. Total chunks created: {len(chunks)}")

# 3. Embed the chunks and store in ChromaDB
print("3. Loading embedding model (this might take a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
client = chromadb.Client()
collection = client.create_collection(name="my_docs")

# We have to give each chunk an ID and the text itself
ids = [str(i) for i in range(len(chunks))]
collection.add(
    documents=chunks,
    ids=ids
)

# 4. Ask a question
question = "What is this document about?" # You can change this!
print(f"\n4. Question: {question}")

# Embed the question and search the database
results = collection.query(
    query_texts=[question],
    n_results=2 
)

context = "\n---\n".join(results['documents'][0])
print(f"\n5. Retrieved Context:\n{context}\n")

# 5. Send Context + Question to Ollama
print("6. Generating answer with Qwen...")
prompt = f"""Answer the question based ONLY on the following context:
{context}

Question: {question}
Answer:"""

payload = {
    "model": "qwen2.5",
    "prompt": prompt,
    "stream": False
}

resp = requests.post("http://localhost:11434/api/generate", json=payload)
print("\n--- FINAL ANSWER ---")
print(resp.json()["response"])