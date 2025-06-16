# memory_retriever.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MemoryRetriever:
    def __init__(self, memory_path):
        self.memory_path = memory_path
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # รองรับภาษาไทย
        self.index = None
        self.texts = []

    def load_memory(self):
        with open(self.memory_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.texts = [entry['content'] for entry in data if 'content' in entry]

    def build_index(self):
        embeddings = self.model.encode(self.texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        return [self.texts[i] for i in indices[0]]

# ======= ตัวอย่างการใช้งาน =======
if __name__ == "__main__":
    retriever = MemoryRetriever(
        r"C:\Users\WithYou\OneDrive\Desktop\heart_of_melah\memory\memory.json"
    )
    retriever.load_memory()
    retriever.build_index()

    results = retriever.search("เมล่ากลัวการถูกลืม")
    for i, result in enumerate(results):
        print(f"[{i+1}] {result}\n")
