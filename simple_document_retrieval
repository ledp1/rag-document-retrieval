"""
Simple RAG (Retrieval-Augmented Generation) Implementation

This approach demonstrates the usefulness of RAG over naive LLM generation by:

1. **Contextual Accuracy**: Instead of relying solely on the LLM's training data,
   RAG retrieves relevant documents from a knowledge base, ensuring answers are
   grounded in specific, up-to-date information.

2. **Reduced Hallucination**: By providing relevant context, RAG significantly
   reduces the likelihood of the LLM generating factually incorrect information
   that might not exist in its training data.

3. **Domain-Specific Knowledge**: The retrieval mechanism allows the system to
   access specialized knowledge (like Project Chimera details) that may not be
   well-represented in the LLM's training corpus.

4. **Transparency**: Users can see which documents informed the answer, providing
   traceability and allowing verification of the information sources.

5. **Scalability**: The knowledge base can be easily updated with new documents
   without retraining the LLM, making it practical for dynamic information systems.

This implementation uses a simple word-overlap scoring mechanism for retrieval,
which while basic, effectively demonstrates the core RAG concept.
"""

from scripts.llm import get_llm_response

KNOWLEDGE_BASE = {
    "doc1": {
        "title": "Project Chimera Overview",
        "content": (
            "Project Chimera is a research initiative focused on developing "
            "novel bio-integrated interfaces. It aims to merge biological "
            "systems with advanced computing technologies."
        )
    },
    "doc2": {
        "title": "Chimera's Neural Interface",
        "content": (
            "The core component of Project Chimera is a neural interface "
            "that allows for bidirectional communication between the brain "
            "and external devices. This interface uses biocompatible "
            "nanomaterials."
        )
    },
    "doc3": {
        "title": "Applications of Chimera",
        "content": (
            "Potential applications of Project Chimera include advanced "
            "prosthetics, treatment of neurological disorders, and enhanced "
            "human-computer interaction. Ethical considerations are paramount."
        )
    }
}


def naive_generation(query):
    prompt = f"Answer directly the following query: {query}"
    return get_llm_response(prompt)


def rag_retrieval(query, documents, min_score=1):
    # Split the query into lowercase words and store them in a set
    query_words = set(query.lower().split())
    matching_docs = []
    
    for doc_id, doc in documents.items():
        # Get content and title words in lowercase
        content_words = set(doc["content"].lower().split())
        title_words = set(doc["title"].lower().split())
        
        # Calculate overlap with content and title separately
        content_overlap = len(query_words.intersection(content_words))
        title_overlap = len(query_words.intersection(title_words))
        
        # Give more weight to title matches (2x) as they're often more relevant
        total_score = content_overlap + (title_overlap * 2)
        
        if total_score >= min_score:
            matching_docs.append((doc, total_score))
    
    # Sort by score in descending order
    matching_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in matching_docs]


def rag_generation(query, documents):
    if not documents:
        prompt = f"No relevant information found. Answer directly: {query}"
    else:
        context = "\n".join([f"- {doc['title']}: {doc['content']}" for doc in documents])
        prompt = (
            f"Use the following information to answer the query. "
            f"If the information is not sufficient, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}"
        )
    return get_llm_response(prompt)


if __name__ == "__main__":
    query = "What is the main goal of Project Chimera?"
    print("Naive approach:", naive_generation(query))
    
    # Get all documents with at least 1 word match
    matching_docs = rag_retrieval(query, KNOWLEDGE_BASE, min_score=1)
    print(f"\nFound {len(matching_docs)} relevant documents")
    for i, doc in enumerate(matching_docs, 1):
        print(f"\nDocument {i} (Title: {doc['title']}):")
        print(doc['content'])
    
    print("\nRAG approach:")
    print(rag_generation(query, matching_docs))