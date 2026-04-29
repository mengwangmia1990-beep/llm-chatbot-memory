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
- Retrieval-Augmented Generation (RAG) with keyword-based retrieval
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

#### Gating Behavior

The current system uses the top1 retrieval score to determine whether to trigger RAG.  
This design simplifies routing but introduces sensitivity to ranking quality.

In practice, this may lead to **gating false negatives**, where relevant queries fail to trigger RAG due to low top1 scores, even when the correct chunk exists in topk results.

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

---

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
### 9. Evaluation - Failure Analysis Breakdown
Below are a few interesting and valuable result data that reflects the potential failures of current system:  
1. This failure type is **should_answer_but_abstained**, given by gating, top1 and topk_recall are all correct. Which means this could be LLM issue.
```json
{
  "query": "can customer return items without a receipt ?", 
  "expected": {
    "answerable_from_kb": true, 
    "expected_use_rag": true, 
    "should_answer_from_kb": true, 
    "should_abstain": false
    }, 
  "actual": {
    "actual_use_rag": true, 
    "abstained": true
    }, 
  "result": {
    "gating_correct": true, 
    "top1_correct": true, 
    "topk_recall": true,  
    "should_abstain_but_answered": false, 
    "should_answer_but_abstained": true
    },
  "generation": {
    "answer_correct": false
    },
  "failure_type": "should_answer_but_abstained"
}
```

2. This question follows a question **"what time does store open on Sunday ?"**. Given by the context, LLM should understand that user is asking the open time of the store on Satuarday. However, due to the keyword overlap matching limitation in the current RAG retrieving logic, system failed to determine the relevancy of this question with knowledge base. Gating failed, top1 and topk_recall both failed, thus fallback to LLM mode. This is a **gating_false_negative** failure.  
**Short or misspelled queries can cause keyword-based retrieval scores to drop below the gating threshold, leading to false negatives even when the query is answerable from the KB.**
```json
{
  "query": "what about Satuarday ?", 
  "expected": {
    "answerable_from_kb": true, 
    "expected_use_rag": true, 
    "should_answer_from_kb": true, 
    "should_abstain": false
    }, 
  "actual": {
    "actual_use_rag": false, 
    "abstained": false
    }, 
  "result": {
    "gating_correct": false, 
    "top1_correct": false, 
    "topk_recall": false, 
    "should_abstain_but_answered": false, 
    "should_answer_but_abstained": false
    }, 
  "generation": {
    "answer_correct": false
    }, 
  "failure_type": "gating_false_negative"
}

```
3. This example falls into **top1_ranking_failure** issue. Although top1 chunk is not the gold chunk, however, topk_recall is true, which means gold chunk is retrieved within topk chunks, therefore LLM is able to provide the correct answer.
```json
{
  "query": "how much discount can a member get ?",
  "expected": {
    "answerable_from_kb": true, 
    "expected_use_rag": true, 
    "should_answer_from_kb": true, 
    "should_abstain": false
    }, 
  "actual": {
    "actual_use_rag": true, 
    "abstained": false
    }, 
  "result": {
    "gating_correct": true, 
    "top1_correct": false, 
    "topk_recall": true,
    "should_abstain_but_answered": false, 
    "should_answer_but_abstained": false
    }, 
  "generation": {
    "answer_correct": null
    }, 
  "failure_type": "top1_ranking_failed"
}

```
4. This question is a out-of-scope query. There is no gold chunk. Therefore the top1_correct and topk_recall are set to None. RAG retrieves the incorrect chunks, however, due to the low score, gating succcessfully rejected it. In other words, retrieval produced unuseful evidence, but gating correctly rejected it.
```json
{
  "query": "what products do you sell ?", 
  "expected": {
    "answerable_from_kb": false, 
    "expected_use_rag": false, 
    "should_answer_from_kb": false,
    "should_abstain": false
    }, 
  "actual": {
    "actual_use_rag": false, 
    "abstained": false
    }, 
  "result": {
    "gating_correct": true, 
    "top1_correct": null, 
    "topk_recall": null, 
    "should_abstain_but_answered": false,  
    "should_answer_but_abstained": false  
    }, 
  "generation": {
    "answer_correct": null
    }, 
  "failure_type": "none"
}
```
5. This question falls into **gating_false_negative** failure. But it reflects a very critical and interesting issue within the current system. Notice topk_recall is true, however, the top1 chunk is false. And actual_use_rag is set to false. The reason is that current RAG uses only top1 score to compare the threshold when gating. **Some false negatives occur even when top-k recall succeeds, because the gating decision depends only on the top-1 score. This suggests improving the gating strategy or using semantic retrieval.**
```json
{
  "query": "how long does delivery usually take ?", 
  "expected": {
    "answerable_from_kb": true, 
    "expected_use_rag": true, 
    "should_answer_from_kb": true, 
    "should_abstain": false  
    }, 
  "actual": {
    "actual_use_rag": false, 
    "abstained": false
    }, 
  "result": {
    "gating_correct": false, 
    "top1_correct": false, 
    "topk_recall": true, 
    "should_abstain_but_answered": false, 
    "should_answer_but_abstained": false  
    }, 
  "generation": {
    "answer_correct": null
    }, 
  "failure_type": "gating_false_negative"
}

```
6. This is a typical **hallucination problem**. Question is not answerable from the knowledge base however the system falls back to LLM model (`actual_use_rag == False`) and LLM hallucinated.
```json
{
  "query": "is VIP customer eligible for free returns ?", 
  "expected": {
    "answerable_from_kb": false, 
    "expected_use_rag": true, 
    "should_answer_from_kb": false, 
    "should_abstain": true
    }, 
  "actual": {
    "actual_use_rag": false, 
    "abstained": false
    }, 
  "result": {
    "gating_correct": false, 
    "top1_correct": null, 
    "topk_recall": null, 
    "should_abstain_but_answered": true, 
    "should_answer_but_abstained": false
    }, 
  "generation": {
    "answer_correct": null
    }, 
  "failure_type": ["gating_false_negative", "should_abstain_but_answered"]
}

```
---
## Evaluation - Summary Insight
From below summary data, we could find valuable insights of current system potential issues:
1. The system suffers from **gating false negatives**, where relevant queries fail to trigger retrieval due to low keyword matching scores.  
2. Topk retrieval recall is relatively strong, indicating that the knowledge base contains sufficient information and can be retrieved when triggered.  
3. Top1 chunk ranking issues reduce the effectiveness of retrieval, as relevant chunks are often not ranked at the top.  
4. The system effectively avoids hallucination in most cases by abstaining when information is insufficient.


Based on the evaluation, the main bottleneck is **keyword-based retrieval** and **gating sensitivity**.  In the next iteration, I plan to introduce embedding-based retrieval to *improve semantic matching* and reduce gating false negatives.

```json
{
  "total_cases": 22, 
  "counts": {
    "gating_correct": 16, 
    "top1_correct": 10, 
    "top1_valid": 16, 
    "topk_recall": 13, 
    "topk_valid": 16, 
    "should_abstain": 5, 
    "correct_abstain": 4, 
    "should_answer": 16, 
    "false_abstain": 1, 
    "hallucination": 1
    }, 
  "rates": {
    "gating_accuracy": 0.7273, 
    "top1_accuracy": 0.625, 
    "topk_recall_rate": 0.8125, 
    "correct_abstain_rate": 0.8, 
    "false_abstain_rate": 0.0625, 
    "hallucination_rate": 0.2
    }, 
  "failure_counts": {
    "gating_false_negative": 6, 
    "topk_recall_failed": 3, 
    "top1_ranking_failed": 6, 
    "should_answer_but_abstained": 1, 
    "should_abstain_but_answered": 1
    }
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

- Replace keyword retrieval with embedding-based retrieval  
- Add reranking for improved relevance  
- Introduce LLM-based evaluation (LLM-as-judge)  
- Improve safety filtering with LLM  
- Add structured output (JSON schema)  
- Build a UI (e.g., Streamlit)  
