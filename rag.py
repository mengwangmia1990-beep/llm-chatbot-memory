# rag.py
import config
import embedding_utils
import math

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
    vectors = []
    if knowledge and len(knowledge) > 0:
        for chunk in knowledge:
            vector = embedding_utils.embed_text(chunk)
            if vector and len(vector) > 0:
                vectors.append({
                    "vector": vector,
                    "chunk": chunk
                })

    return vectors


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def retrieve_embedding(query, embedded_knowledge, top_k=2, gating=0.0):
    query_vector = embedding_utils.embed_text(query)
    scored_chunks = []
    
    if query_vector and len(query_vector) > 0:
        for doc in embedded_knowledge:
            score = cosine_similarity(query_vector, doc["vector"])
            scored_chunks.append({
                "chunk": doc["chunk"],
                "score": score
            })

    # sort score by decreasing order
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # select topk
    candidate_chunks = scored_chunks[:top_k]

    # Threshold Gating
    top_scores = [item["score"] for item in candidate_chunks]
    top_chunks = [item["chunk"] for item in candidate_chunks]

    top1_chunk = top_chunks[0] if top_chunks else []
    top2_chunk = top_chunks[1] if top_chunks and len(top_chunks) > 1 else []

    top1_score = top_scores[0] if top_scores else 0.0
    top2_score = top_scores[1] if top_scores and len(top_scores) > 1 else 0.0
    top1_top2_gap = top1_score - top2_score

    gating_threshold = gating if gating != 0.0 else config.RAG_RELEVANCE_EMBEDDING_THRESHOLD
    use_rag = top1_score >= gating_threshold

    return {
        "top_chunks": top_chunks,
        "top_scores": top_scores,

        "top1_chunk": top1_chunk,
        "top2_chunk": top2_chunk,

        "top1_score": top1_score,
        "top2_score": top2_score,
        
        "use_rag": use_rag
    }