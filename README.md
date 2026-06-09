# RAG Chatbot with Grounding Validation, Memory & Evaluation Framework

## Overview
Built a production-minded RAG chatbot with hybrid memory, grounding validation, and evaluation pipelines. 

The system evolved through three major iterations: keyword retrieval, embedding retrieval, and grounding-aware generation. Evaluation pipelines were used throughout the project to analyze retrieval quality, answerability, abstention behavior, and hallucination rates. 

### Key Findings
- Embedding retrieval significantly outperformed keyword retrieval.
- Retrieval relevance does not imply answerability.
- Higher retrieval thresholds reduced retrieval coverage and increased hallucination through LLM fallback.
- Answerability and answer grounding are distinct problems that require separate evaluation.

**A central insight from this project is that retrieval relevance and answerability are fundamentally different concepts.** A retrieved document may be semantically relevant to a query while still lacking sufficient information to answer it faithfully.

### Architecture Diagram
```text
User Query
     │
     ▼
 Retrieval
     │
     ▼
 Grounding Validation
     │
     ▼
 Answerable?
 ┌───┴────┐
 │        │
Yes       No
 │        │
 ▼        ▼
 RAG     I don't know
     │
     ▼
  Response
     │
     ▼
 Runtime Trace

──────────────────
Golden Dataset
      │
      ▼
 Evaluation Pipeline
      │
      ▼
 Failure Analysis
```

## Motivation
Traditional RAG systems typically assume that retrieving relevant documents is sufficient for generating correct answers.

However, during development I observed that retrieval relevance and answerability are fundamentally different concepts. A retrieved document may be semantically relevant to a query while still lacking enough information to answer it faithfully.  

For example:

```text
Question:
Do non-members get a 10% discount?

Retrieved Document:
Members get a 10% discount.

Expected Answer:
I don't know.
```

The retrieved document is highly relevant. However, it does not contain enough information to answer the user's question.  

LLM may incorrectly infer that non-members receive no discount, even though this information is not present in the knowledge base.

To better understand this problem, the system evolved through multiple retrieval and grounding strategies, each evaluated using a dedicated benchmarking pipeline.

## System Evolution
### Iteration I - Keyword-Based Retrieval
The first version of the system used a simple keyword-overlap retrieval strategy.  

Query keywords and document keywords were compared using overlap ratios, and the highest-ranked chunks were passed to the language model as retrieval context.  

```text
User Query 
    ↓ 
Keyword Matching 
    ↓ 
Top-K Chunks 
    ↓ 
LLM Response
```
A retrieval threshold determined whether retrieved knowledge was sufficiently relevant. If the threshold was not met, the system fell back to standard LLM generation.  

### Findings
Keyword retrieval served as a useful baseline due to its simplicity and interpretability. However, it struggled with typos, paraphrasing, and semantically similar questions that shared few keywords with the knowledge base.

These limitations motivated the next iteration: embedding-based retrieval.

---

### Iteration II - Embedding-Based Retrieval
To improve retrieval quality, the second iteration replaced keyword matching with embedding-based semantic retrieval.  

Knowledge chunks were converted into vector embeddings. For each user query, an embedding was generated and compared against all knowledge vectors using cosine similarity.  

The system ranked documents by semantic similarity and returned the top-k chunks as retrieval context.  

A configurable similarity gating threshold was used to determine whether retrieved knowledge should be included in generation.  

Gating threshold tuning revealed an important trade-off: **overly strict thresholds reduced retrieval coverage and increased hallucinations** through LLM fallback generation. 

```text
User Query
    ↓
Embedding
    ↓
Cosine Similarity
    ↓
Top-K Chunks
    ↓
Gating Threshold
       │
 ┌─────┴─────┐
 │           │
Use RAG?     No
 │           │
 ▼           ▼
RAG      LLM Only
```

### Evaluation
To measure the impact of semantic retrieval, a golden evaluation dataset was created to compare keyword retrieval and embedding retrieval.  

Metrics included:
- Top-1 Retrieval Accuracy
- Top-k Recall
- Gating Accuracy

The evaluation framework enabled per-case analysis and failure categorization, making retrieval behavior observable beyond final answer quality.

### Findings
Embedding retrieval significantly outperformed keyword retrieval across retrieval accuracy and recall metrics. Semantic retrieval achieved nearly 100% Top-1 retrieval accuracy and Top-K recall.

Semantic retrieval was more robust to:

- Paraphrased questions
- Different wording
- Minor spelling variations

However, a new problem emerged during evaluation:  

Although embedding retrieval successfully identified semantically relevant documents, some retrieved documents still lacked sufficient information to answer the user's question.  
 
This observation revealed a key limitation of retrieval-based systems: **retrieval relevance does not necessarily imply answerability**. This insight motivated the next iteration: grounding validation.

---

### Iteration III - Grounding Validation
To address the answerability gap discovered in Iteration II, a grounding validation layer was introduced between retrieval and generation.

### Design
The grounding validator determines whether the retrieved context contains sufficient evidence to answer the user's question.

Only answerable queries proceed to generation. Otherwise, falls back to LLM generation.

```text
User Query 
    ↓ 
Embedding Retrieval 
    ↓ 
Grounding Validation 
    ↓ 
Answerable? 
┌───┴────┐ 
│        │ 
Yes      No 
│        │ 
▼        ▼ 
Answer  LLM Generation
```

### Evaluation
Grounding validation was evaluated against direct generation using:  
- Grounding Accuracy
- Unsupported Answer Rate
- False Abstention Rate

### Findings
The initial grounding validation implementation did not behave as expected. While grounding validation reduced false abstentions, it also increased unsupported answers in some cases.  

```text
Example:

Query:
Can customers return broken items?

Retrieved Document:
(return policy does not mention broken items)

Expected:
I don't know.

Actual:
Customers cannot return broken items because the return policy does not mention returning broken items.
```
The evaluation revealed that unsupported queries could still reach the language model through fallback generation, leading to unsupported answers.

In addition, the grounding validator sometimes treated missing information as evidence for a negative answer.

> Not mentioned != Not Allowed

This failure revealed that a retrieved document can be highly relevant to a query while still lacking enough information to answer it faithfully.  

The issue motivated the next iteration: grounding prompt refinement.

---

### Iteration III.1 - Grounding Prompt Refinement
The evaluation from Iteration III revealed that the grounding validator sometimes treated missing information as evidence for a negative answer.  

To address this issue, the grounding prompt was refined to enforce a stricter definition of answerability.  

### Design
The updated grounding validator was instructed to:

- identify information explicitly supported by the retrieved context
- distinguish unsupported questions from negative answers
- avoid inferring unstated policies or business rules
- treat missing information as unanswerable rather than false

A key rule added to the prompt was:

**The answer is answerable only if the context explicitly provides the requested information or explicitly contradicts the user's premise.** 

In addition, a hard gating mechanism was introduced.  

If the grounding validator determined that a question was not answerable from the retrieved context, the system immediately returned: I don't know, instead of allowing fallback LLM generation.

This design prevents the language model from generating unsupported conclusions when sufficient evidence is unavailable.

```text
User Query 
    ↓ 
Embedding Retrieval 
    ↓ 
Grounding Validation 
    ↓ 
Answerable? 
┌───┴────┐ 
│        │ 
Yes      No 
│        │ 
▼        ▼ 
Answer  I don't know
```

###  Evaluation
The updated grounding validator was evaluated using the same benchmark dataset and same metrics from Iteration III.  

### Findings
Prompt refinement successfully distinguished answerable and unanswerable queries on the evaluation dataset.  

Compared with direct generation, grounding validation provided two major benefits:

- The grounding validator became more conservative when handling missing information and was less likely to interpret "not mentioned" as a negative answer.
- Reduced false abstentions by helping the model recognize when retrieved context contained sufficient information to answer the question.

```text
Example 1:  
query: Do non-members get a 10% discount?
document: Members get a 10% discount.
```

Without grounding, the model incorrectly inferred that non-members do not receive the discount, even though the retrieved document never stated this information. With grounding validation, the system correctly abstained because the retrieved evidence did not answer the question.

```text
Example 2:
query: can customer return items without a receipt?
document: Customers can return items within 7 days with a receipt. 
```
Without grounding, the model abstained despite having sufficient evidence to answer the question. With grounding validation, the system correctly identified the query as answerable and generated a supported answer.  

### Conclusion

These results reinforced the central insight of the project:

**Retrieval relevance does not imply answerability.**

A retrieved document may be highly relevant to a user's question while still lacking sufficient evidence to answer it faithfully. Grounding validation helps bridge this gap by explicitly validating answerability before generation.


## Evaluation Framework
To make system behavior measurable, a dedicated evaluation pipeline was built around golden datasets and per-case evaluation. 

### Design
Each evaluation case defines an expected system behavior, including retrieval relevance, answerability, abstention behavior, and expected responses.  

Each evaluation runner compares actual system outputs against expected results defined in the golden dataset.

Different evaluation runners were created for different experiments, including:

- Keyword Retrieval vs Embedding Retrieval
- Retrieval Threshold Tuning
- Direct Generation vs Grounding Validation

Below is the architecture of an evaluation runner:
```text
Golden Dataset 
      │ 
      ▼ 
generate_reply()
      │ 
      ▼ 
Actual Result 
      │ 
      ▼ 
Per-Case Comparison
      │ 
      ▼ 
Evaluation Summary
```
### Benefits
The evaluation framework made failure modes observable and enabled systematic comparison across different retrieval and grounding strategies. Several key findings throughout the project emerged directly from this framework.


## Hybrid Memory System
To support long-running conversations while keeping context size bounded, a hybrid memory architecture was implemented.  

### Design
The memory system combines:

- Short-Term Memory: the most recent N conversation turns
- Long-Term Memory: an incrementally updated conversation summary

When older conversations fall outside the short-term memory window, a conversation delta is generated and merged into the existing summary.  

This approach keeps the context window size approximately constant regardless of conversation length.

```text
                    Conversation History
                             │
        ┌────────────────────┴───────────────────┐
        │                                        │
        ▼                                        ▼
 Recent N Dialogs                           Older Dialogs
        │                                        │
        ▼                                        ▼
 Short-Term Memory                    Conversation Delta
                                                 │
                                                 ▼
                                          Old Summary
                                                 │
                                                 ▼
                                          Summary Merge
                                                 │
                                                 ▼
                                          Long-Term Memory
                                                 │
                        ┌────────────────────────┘
                        │
                        ▼
                Final Prompt Context
```

### Benefits
The hybrid design enables long-running conversations while keeping context size bounded.

By combining recent dialogue history with a compact long-term summary, the system can retain important conversational context without unbounded growth.

### Limitation
The primary trade-off is *Information loss*. As conversations are repeatedly compressed into summaries, details may gradually disappear or become less precise over time.  

This represents a common trade-off in summary-based memory systems: **the ability to support longer conversations comes at the cost of gradual information loss**.


## Future Improvements
### Retrieval
- Add reranking to improve top-1 relevance
- Introduce vector database for larger knowledge bases

### Grounding & Answerability
- Separate answerability detection from answer grounding so the system first determines whether the retrieved context contains enough information to answer, then independently verifies whether the generated answer is fully supported by evidence.

### Memory
- Reduce information loss in long-term summaries
- Explore retrieval-augmented memory for preserving detailed historical context

### Productization
- Build a Streamlit UI
- Add persistent storage and user profiles