import faiss
import pickle
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_PATH = "vectorstore"


# -------- LOAD EMBEDDING MODEL --------
print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -------- LOAD VECTOR DB --------
print("Loading vector database...")

index = faiss.read_index(f"{VECTOR_PATH}/index.faiss")

with open(f"{VECTOR_PATH}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


# -------- RETRIEVAL FUNCTION --------
def retrieve(query, top_k=3):
    query_vector = embed_model.encode([query])

    distances, indices = index.search(np.array(query_vector), top_k)

    results = [chunks[i] for i in indices[0]]

    return results


# -------- PROMPT TEMPLATE --------
def build_prompt(query, context):
    prompt = f"""
You are an Indian legal assistant AI.

Use ONLY the provided legal context to answer.

Follow STRICTLY this format:

1. Relevant Law / Section
2. Explanation in simple language
3. Example scenario
4. Possible legal actions
5. Advice / next steps
6. Disclaimer

Context:
{context}

User Question:
{query}

Answer:
"""
    return prompt


# -------- CHAT FUNCTION --------
def chat(query):

    retrieved_chunks = retrieve(query)

    context = "\n\n".join(retrieved_chunks)

    prompt = build_prompt(query, context)

    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# -------- TEST --------
if __name__ == "__main__":

    while True:
        q = input("\nAsk a legal question: ")

        if q.lower() == "exit":
            break

        answer = chat(q)

        print("\n", answer)