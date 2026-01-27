import os
import json
from typing import List, Dict, Any, Optional
import math
from collections import defaultdict


class VectorStore:
    """
    Simple in-memory vector store for RAG.
    Uses TF-IDF similarity for document retrieval without external dependencies.
    """
    
    def __init__(self, db_path: str = "chroma_db"):
        """Initialize in-memory vector store"""
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # In-memory storage
        self.documents = {}  # id -> document text
        self.metadata = {}   # id -> metadata dict
        self.vocabulary = set()  # All unique words
        self.doc_frequencies = defaultdict(lambda: defaultdict(int))  # word -> {doc_id -> count}
        self.persistence_file = os.path.join(db_path, "documents.json")
        
        # Load from disk if exists
        self._load_from_disk()

    def add_documents(self, documents: List[str], ids: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add documents to the vector store"""
        if metadata is None:
            metadata = [{"source": "document"} for _ in documents]
        
        for doc, doc_id, doc_meta in zip(documents, ids, metadata):
            # Store document and metadata
            self.documents[doc_id] = doc
            self.metadata[doc_id] = doc_meta
            
            # Update vocabulary and frequencies
            words = self._tokenize(doc)
            word_counts = defaultdict(int)
            for word in words:
                self.vocabulary.add(word)
                word_counts[word] += 1
                self.doc_frequencies[word][doc_id] = word_counts[word]
        
        # Persist to disk
        self._save_to_disk()

    def retrieve(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Retrieve relevant documents using TF-IDF similarity"""
        if not self.documents:
            return {"documents": [[]], "ids": [[]], "distances": [[]]}
        
        # Tokenize query
        query_words = self._tokenize(query)
        
        # Calculate TF-IDF scores for each document
        scores = {}
        for doc_id in self.documents:
            score = 0.0
            for word in query_words:
                if word in self.doc_frequencies:
                    # TF: term frequency in document
                    tf = self.doc_frequencies[word].get(doc_id, 0) / (len(self._tokenize(self.documents[doc_id])) or 1)
                    # IDF: inverse document frequency
                    idf = math.log(len(self.documents) / (len(self.doc_frequencies[word]) + 1))
                    score += tf * idf
            scores[doc_id] = score
        
        # Get top n results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        # Format results to match chromadb format
        doc_ids = [doc_id for doc_id, _ in sorted_results]
        doc_texts = [self.documents[doc_id] for doc_id, _ in sorted_results]
        distances = [1.0 - (score / (max(scores.values()) or 1)) for _, score in sorted_results]
        
        return {
            "documents": [doc_texts],
            "ids": [doc_ids],
            "distances": [distances]
        }

    def format_retrieved_docs(self, results: Dict[str, Any]) -> str:
        """Format retrieved documents for LLM context"""
        if not results or not results.get("documents") or not results["documents"][0]:
            return ""
        
        formatted = "Retrieved Context:\n"
        for i, doc in enumerate(results["documents"][0], 1):
            formatted += f"\n[Document {i}]:\n{doc}\n"
        
        return formatted

    def delete_all(self) -> None:
        """Clear all documents"""
        self.documents = {}
        self.metadata = {}
        self.vocabulary = set()
        self.doc_frequencies = defaultdict(lambda: defaultdict(int))
        self._save_to_disk()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split by whitespace, remove punctuation"""
        import string
        text = text.lower()
        # Remove punctuation and split
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        words = [w for w in text.split() if w]
        return words

    def _save_to_disk(self) -> None:
        """Persist documents to disk"""
        data = {
            "documents": self.documents,
            "metadata": self.metadata
        }
        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self) -> None:
        """Load documents from disk if they exist"""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", {})
                    self.metadata = data.get("metadata", {})
                    
                    # Rebuild vocabulary and frequencies
                    for doc_id, doc_text in self.documents.items():
                        words = self._tokenize(doc_text)
                        word_counts = defaultdict(int)
                        for word in words:
                            self.vocabulary.add(word)
                            word_counts[word] += 1
                            self.doc_frequencies[word][doc_id] = word_counts[word]
            except Exception as e:
                print(f"Warning: Could not load documents from disk: {e}")
