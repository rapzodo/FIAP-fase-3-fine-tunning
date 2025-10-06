import chromadb
from sentence_transformers import SentenceTransformer


class AmazonProductRAG:
    """RAG system for Amazon product data using ChromaDB and semantic search"""

    def __init__(self, db_path="./chroma_db", collection_name="amazon_products"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} items")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Amazon product titles and descriptions"}
            )
            print(f"Created new collection '{collection_name}'")

    def index_training_data(self, training_data):
        """Index training data into ChromaDB for fast semantic search"""
        if self.collection.count() > 0:
            print(f"Collection already has {self.collection.count()} items. Skipping indexing.")
            return

        print(f"Indexing {len(training_data)} products into vector database...")

        documents = []
        metadatas = []
        ids = []

        for idx, item in enumerate(training_data):
            title = item['input']
            description = item['output']

            if not description or not description.strip():
                continue

            # Create a rich document combining title and description
            doc_text = f"Product: {title}\nDescription: {description}"

            documents.append(doc_text)
            metadatas.append({
                "title": title,
                "description": description
            })
            ids.append(f"product_{idx}")

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

        print(f"✅ Indexed {len(documents)} products successfully!")

    def find_relevant_references(self, query, top_k=1):
        """Find most relevant products using semantic search"""
        if self.collection.count() == 0:
            print("⚠️ No data in collection. Please index training data first.")
            return []

        # Query the vector database
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        # Format results
        references = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for metadata in results['metadatas'][0]:
                references.append({
                    'input': metadata['title'],
                    'output': metadata['description']
                })

        return references

    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Amazon product titles and descriptions"}
            )
            print(f"✅ Cleared collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error clearing collection: {e}")


# Singleton instance
_rag_instance = None

def get_rag_instance(db_path="./chroma_db"):
    """Get or create the RAG singleton instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = AmazonProductRAG(db_path=db_path)
    return _rag_instance

