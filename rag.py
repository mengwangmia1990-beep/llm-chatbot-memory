# rag.py
import config

# Load and clean knowledge data
def load_knowledge(filepath="knowledge.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    knowledge = []
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            knowledge.append(cleaned)
    return knowledge


def calculate_score(query_words, chunk_words):
    # scoring strategy: count query word overlap in chunk
    query_set = set(query_words)
    chunk_set = set(chunk_words)

    if not query_set:
        return 0.0
    
    overlap = query_set & chunk_set
    overlap_ratio = len(overlap) / len(query_set)

    return overlap_ratio


def retrieve_keyword(query, knowledge, top_k=2):
    query_words = query.lower().split() # query preprocessing
    scored_chunks = []

    for chunk in knowledge:
        chunk_words = chunk.lower().split() # chunk preprocessing
        chunk_score = calculate_score(query_words, chunk_words)
        scored_chunks.append((chunk, chunk_score))

    # Sort
    scored_chunks.sort(reverse=True, key=lambda x: x[1])

    # Select top_k chunks
    candidate_chunks = scored_chunks[:top_k]

    # Threshold Gating
    top_scores = [score for chunk, score in candidate_chunks]
    top_chunks = [chunk for chunk, score in candidate_chunks]

    if top_scores:
        top_score = top_scores[0]
    else:
        top_score = 0
    
    use_rag = False
    if top_scores and top_scores[0] >= config.RAG_RELEVANCE_THRESHOLD:
        use_rag = True

    return {
        "top_chunks": top_chunks,
        "top_scores": top_scores,
        "top_score": top_score,
        "use_rag": use_rag # use_rag indicates that the system considers the knowledge base relevant to the query, 
# and therefore includes retrieved context to assist the LLM in answering.
    }


def embed_knowledge(knowledge):
    return None


def retrieve_embedding(query, embedded_knowledge, top_k=2):
    return None