# AI Conversational Assistant with Memory, Safety & Hybrid RAG

## Overview
This project is a CLI-based conversational assistant powered by the OpenAI API.

It supports multi-turn dialogue with hybrid memory (short-term + summarized long-term), 
a lightweight Retrieval-Augmented Generation (RAG) pipeline, and a structured evaluation framework for analyzing system performance.

The system follows a **hybrid answering strategy**:
- When relevant knowledge is retrieved, responses are grounded using RAG
- Otherwise, the system falls back to the base LLM

In addition, the project includes an **evaluation pipeline** that measures system performance across routing, retrieval, and generation, enabling systematic failure analysis and iterative improvement.

---

## Key Features

- Multi-turn conversation with LLM
- Hybrid memory (short-term + long-term summarized memory)
- Retrieval-Augmented Generation (RAG) with keyword-based and embedding-based retrieval (iteration II)  
- Threshold-based RAG gating (dynamic routing between RAG and LLM)
- Hybrid answering strategy (grounded + fallback responses)
- Rule-based safety filtering
- Observability via structured trace logging (JSONL)

### Evaluation & Analysis
- Structured evaluation dataset with labeled ground truth
- Per-case evaluation (routing, retrieval, generation)
- Failure taxonomy (gating, retrieval, abstention, hallucination)
- Aggregate metrics (accuracy, recall, abstain behavior)
- Failure breakdown analysis for system debugging

---

## System Design

### 1. Chat Message Schema

```json
{
  "role": "system",
  "content": "Instructions, rules, and context"
},
{
  "role": "system",
  "content": "Summarized long-term conversation memory"
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

- The first system message defines the assistant’s global behavior, safety rules, and response constraints.  
- The second system message stores summarized long-term memory from earlier conversation turns.  
- Recent user/assistant messages are kept as short-term memory for local conversational context.

---

### 2. Conversation Flow

User Input  
→ Input Validation & Safety Filtering  
→ Retrieval & Gating (routing RAG or LLM path)  
→ Prompt Construction (inject RAG context if applicable)  
→ LLM Inference  
→ Update Conversation State (append user & assistant messages)    
→ Context Window Management  
→ Optional Summarization  
→ Observability Logging (structured trace)  

---

### 3. Retrieval-Augmented Generation (RAG)

The system supports two retrieval strategies:

- **Keyword-based retrieval (Iteration I)**
  - Uses token overlap for scoring
  - Sensitive to query phrasing, typos, and lexical variation

- **Embedding-based retrieval (Iteration II)**
  - Uses semantic similarity (cosine similarity on embeddings)
  - More robust to paraphrasing and natural language variation

Both strategies share a unified RAG pipeline:

- Top-k chunks are retrieved from the knowledge base  
- A relevance gating mechanism determines whether to trigger RAG  
- Retrieved context is injected into the prompt when RAG is enabled  

---

#### Key Insight

**Relevancy and answerable are fundementally different concepts**

Evaluation across iterations reveals three distinct stages:

```text
relevance → retrieval → answerability
```

---

### 4. Hybrid Answering Strategy

The system dynamically routes responses based on retrieval confidence:

- **RAG mode (`mode="rag"`)**
  - Triggered when retrieval score exceeds threshold
  - Provides grounded responses using external knowledge

- **Fallback mode (`mode="llm"`)**
  - Triggered when retrieval confidence is low
  - Allows general reasoning but may introduce hallucination risk

---

### 5. Memory Management

#### Hybrid Memory Design

- Short-term memory: recent conversation messages  
- Long-term memory: summarized history stored as system message  

#### Summarization Strategy

- Triggered when context window length exceeds threshold  
- Only intermediate messages are summarized  
- Most recent messages are preserved for recency  
- **Fallback**: if summarization fails, the system degrades to keeping only the most recent messages

#### Optimization

- Limit repeated summarization of existing summaries to mitigate cumulative information loss  
- Preserve important context while reducing token usage  

---

### 6. Failure Handling

- LLM API failure → continue chat loop  
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
    "top_chunk": ...
    "top_scores": [...],
    "top_score": ...
    "threshold": ...
  },
  "response": {
    "reply": "...",
    "mode": "rag"
  },
  "error": "..."
}
```

This enables:
- Debugging retrieval quality  
- Analyzing model behavior  
- Supporting manual and automated evaluation  

---

### 8. Evaluation - Pipeline Workflow  
→ Load evaluation dataset `eval_data.jsonl` (expected behaviors with gold answers, etc)  
→ Generate model response (actual behaviors)  
→ Perform per-case evaluation  
→ Generate evaluation report `eval_report.jsonl` (JSONL with structured metrics and failure types)  
→ Metrics Aggregation  

---
### 9. Evaluation - Failure Analysis

We categorize failures into three stages: **routing, retrieval, and generation**.  
This taxonomy helps isolate failures across different stages of the RAG pipeline, enabling targeted debugging and system improvement.

- **Routing (Gating)**
  - `GATING_FALSE_POSITIVE`: KB is not relevant but RAG is triggered  
  - `GATING_FALSE_NEGATIVE`: KB is relevant but RAG is not triggered  

- **Retrieval (Recall)**
  - `TOPK_RECALL_FAILED`: Gold chunk is not present in top-k results  

- **Grounding / Abstention**
  - `SHOULD_ABSTAIN_BUT_ANSWERED`: Hallucination (answer without support)  
  - `SHOULD_ANSWER_BUT_ABSTAINED`: False abstention (missed answer)  

- **Answer Correctness**
  - `answer_correct`: correctness of final answer (manual labeling; LLM-as-judge planned)

---
### 10. Evaluation - System Evolution

#### Iteration I: Keyword-based Retrieval (Baseline)

Keyword-based retrieval reveals several limitations:  

- Gating false negatives due to low lexical overlap  
- Sensitivity to query phrasing and typos  
- Imperfect **top-k recall** and **top-1 ranking issue**    

Example failure:
- “what about Satuarday?” → gating fails due to spelling variation  
- “how long does delivery usually take?” → top-1 ranking issue

**Conclusion:**
> The main bottleneck in Iteration I is retrieval quality and gating sensitivity.

---

#### Iteration II: Embedding-based Retrieval

Replacing keyword retrieval with embedding-based semantic retrieval leads to:

- Top-1 accuracy ≈ 1.0  
- Top-k recall ≈ 1.0  
- Gating accuracy significantly improved  

This effectively resolves retrieval-related failures.

--- 

#### Embedding-based Retrieval Threshold Tuning

We performed a simple threshold sweep on the embedding-based retrieval system. Gating threshold ranges from [0.3, 0.35, 0.4, 0.45]

Results show:

- Lower threshold 0.3 achieves the best overall performance  
- Higher thresholds introduce more gating false negatives  
- At high threshold 0.45, hallucination starts to appear due to increased fallback to LLM  

Interestingly, false abstention remains constant across thresholds, indicating that:

> the remaining errors are not caused by retrieval or gating, but by generation limitations.

---

**Conclusion:**  
We ended up selecting `gating_threshold = 0.3` as it provides the best tradeoff between recall and hallucination.

---

#### Case review

In this case, the system correctly identifies the relevant knowledge and triggers RAG. However, the model still responds with *"I don't know."*.

This illustrates an important limitation:

> **Relevance does not guarantee answerability.**

Although the retrieved chunk is highly relevant, the answer is not explicitly stated in the knowledge base and requires simple reasoning (e.g., negation). Due to strict grounding constraints, the model abstains instead of answering.

This type of failure motivates the next iteration, where we will introduce **LLM-based answer grounding / answerability checks** to better handle implicitly answerable queries.

```json
{
  "query": "can customer return items without a receipt ?", 
  "retrieval": {
    "mode": "embedding", 
    "threshold": 0.3, 
    "use_rag": true, 
    "top-k": 2, 
    "top_chunks": ["Return Policy: Customers can return items within 7 days with a receipt.", "Business Hours: Monday to Friday open at 9am and close at 8pm. Saturday open at 8am and close at 6pm. Sunday closed."], "top_scores": [0.6268515496496546, 0.23584901986657547], 
    "top1_chunk": "Return Policy: Customers can return items within 7 days with a receipt.", 
    "top1_score": 0.6268515496496546, 
    "top2_chunk": "Business Hours: Monday to Friday open at 9am and close at 8pm. Saturday open at 8am and close at 6pm. Sunday closed.", 
    "top2_score": 0.23584901986657547, 
    "top1_top2_gap": 0.39100252978307914
    }, 
  "response": {
    "reply": "I don't know.", 
    "mode": "rag"
    }, 
  "error": null
}
```
---

## Why This Design?

- Prevents context explosion  
- Controls latency and token cost  
- Improves response grounding  
- Enables system observability and evaluation  
- Supports iterative improvement (RAG, prompt, memory)  

---

## Future Improvements

### Iteration II: Embedding-based RAG
- Replace keyword-based retrieval with embedding-based semantic retrieval
- Compare keyword retrieval vs. embedding retrieval using the existing evaluation pipeline
- Measure improvements in gating accuracy, top-k recall, and false negative rate

### Iteration III: Answer Quality & Grounding
- Introduce LLM-as-judge to evaluate answer correctness and groundedness
- Add structured output using JSON schema
- Add answer verification to reduce unsupported or hallucinated responses

### Additional Improvements
- Improve safety filtering with LLM-based classification
- Add reranking to improve top-1 relevance
- Build a UI, such as Streamlit, for easier interaction and demo