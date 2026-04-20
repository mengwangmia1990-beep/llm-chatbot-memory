# rag.py

# Load and clean knowledge data
def load_knowledge(filepath="knowledge.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 去掉空行和首尾空格
    knowledge = []
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            knowledge.append(cleaned)
    return knowledge


def retrieve(query, knowledge, top_k=2):
    query_words = query.lower().split()
    scored_chunks = []

    for chunk in knowledge:
        score = 0
        chunk_lower = chunk.lower()

        for word in query_words:
            if word in chunk_lower:
                score += 1

        if score > 0:
            scored_chunks.append((score, chunk))

    # 按 score 从大到小排序
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    # 只取前 top_k 条 chunk
    top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
    return top_chunks