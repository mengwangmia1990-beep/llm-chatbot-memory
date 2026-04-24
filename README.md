# AI Conversational Assistant with Memory, Safety & Hybrid RAG

## Overview
This project is a CLI-based conversational assistant powered by the OpenAI API.  
It supports multi-turn dialogue with hybrid memory (short-term + summarized long-term), rule-based safety filtering, and a lightweight Retrieval-Augmented Generation (RAG) pipeline.

The system follows a **hybrid answering strategy**:
- When relevant knowledge is retrieved, responses are grounded using RAG
- Otherwise, the system falls back to the base LLM to provide general answers

---

## Key Features

- Multi-turn conversation with LLM
- Hybrid memory (short-term + long-term summarized memory)
- Context window management with summarization
- Retrieval-Augmented Generation (RAG)
- Conditional knowledge injection (non-persistent)
- Hybrid answering (RAG + fallback LLM)
- Rule-based safety filtering
- Basic failure handling (rollback mechanism)
- Observability via structured trace logging (JSONL)

---

## System Design

### 1. Chat Message Schema

```json
{
  "role": "system",
  "content": "Instructions, rules, and context"
},
{
  "role": "user",
  "content": "User input"
},
{
  "role": "assistant",
  "content": "Model response"
}
```

- The first system message defines global behavior  
- The second system message stores summarized conversation history  

---

### 2. Conversation Flow

User Input  
→ Safety Check  
→ Append to Message History  
→ Retrieval (RAG decision)  
→ Prompt Construction (optional RAG context injection)  
→ LLM Call  
→ Append Assistant Response  
→ Context Window Management  
→ Optional Summarization  

---

### 3. Retrieval-Augmented Generation (RAG)

- Current implementation uses keyword-based scoring (token overlap)
- Top-k chunks are selected and injected into the prompt
- Threshold-based gating determines whether to use RAG

#### Known Limitations

- Ranking instability for short or ambiguous queries  
- Lexical matching struggles with semantic similarity  
- Top-k may include noisy or partially relevant chunks  

#### Key Insight

Even when the correct chunk appears in top-k, the model may:
- Recover the correct answer  
- Fail due to noisy context  
- Or respond conservatively ("I don't know")  

---

### 4. Hybrid Answering Strategy

The system dynamically routes responses:

- **RAG mode (`mode="rag"`)**
  - Triggered when relevant knowledge is detected
  - Uses retrieved context to produce grounded responses

- **Fallback mode (`mode="llm"`)**
  - Triggered when no relevant knowledge is found
  - Uses base LLM to answer general questions

---

### 5. Memory Management

#### Hybrid Memory Design

- Short-term memory: recent conversation messages  
- Long-term memory: summarized history stored as system message  

#### Summarization Strategy

- Triggered when context exceeds threshold  
- Only intermediate messages are summarized  
- Recent messages are preserved for recency  

#### Optimization

- Avoid re-summarizing existing summaries  
- Preserve important context while reducing token usage  

---

### 6. Failure Handling

- API failure → rollback user message  
- Summarization failure → fallback to truncated recent history  

---

### 7. Observability (Trace Logging)

Each interaction is logged as structured JSON (JSONL format):

```json
{
  "query": "...",
  "retrieval": {
    "use_rag": true,
    "top_chunks": [...],
    "top_scores": [...],
    "threshold": ...
  },
  "response": {
    "answer": "...",
    "mode": "rag"
  }
}
```

This enables:

- Debugging retrieval quality  
- Analyzing model behavior  
- Supporting manual and automated evaluation  

### 8. Evaluation - Failure Analysis
We categorize failures into three stages: routing, retrieval, and generation. This taxonomy helps isolate failures across different stages of the RAG pipeline, enabling targeted debugging and system improvement.
- **Routing (Gating)**
  - use_rag = `True` indicating system considers knowledge base relevant to the user query, therefore includes the relevant evidence/context to assit LLM for answering.
  - use_rag = `False` for general questions. 
  - `GATING_FALSE_POSITIVE`: Knowledge base is not relevant to the question, however, system considers the relevancy.
  - `GATING_FALSE_NEGATIVE`: Knowledge base is relevant to the question, however, system denies the relevancy.
- **Retrieval (Recall)**
  - `TOPK_RECALL_FAILED`: Question is **answerable** from knowledge base, system finds knowledge base relevant to the question (use_rag = `True`), however, topk chunks does not contain the **gold_chunk**.
- **Abstain issue (Grounding / Abstention)**
  - `SHOULD_ABSTAIN_BUT_ANSWERED`: Given by the query is **not answerable** from knowledge base, system finds knowledge base relevant (use_rag = `True`) and sends the context to LLM, however LLM ends up answering questions without saying "I don't know".
  - `SHOULD_ANSWERED_BUT_ABSTAIN`: Given by the query is **answerable** from knowledge base, system finds knowledge base relevant (use_rag = `True`) and sends the context to LLM, however, LLM replys "I don't know".  
- **Answer Correctness**
  - `answer_correct`: system provides the correct expected answer. (So far this metric is tagged manually, will use `llm_as_judge` in the future iteration.)



---

## Why This Design?

- Prevents context explosion  
- Controls latency and token cost  
- Improves response grounding  
- Enables system observability and evaluation  
- Supports iterative improvement (RAG, prompt, memory)  

---

## Future Improvements

- Replace keyword retrieval with embedding-based retrieval  
- Add reranking for improved relevance  
- Introduce LLM-based evaluation (LLM-as-judge)  
- Improve safety filtering with LLM  
- Add structured output (JSON schema)  
- Build a UI (e.g., Streamlit)  
