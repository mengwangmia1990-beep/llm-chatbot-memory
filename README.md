# AI Conversational Assistant with Memory & Safety

## Overview
This project is a CLI-based chatbot powered by the OpenAI API. 
It supports multi-turn conversation with short-term and summarized long-term memory, basic rule-based safety filtering, and a retrieval-augmented generation (RAG) pipeline that injects relevant knowledge into the prompt to produce grounded responses.

## Key Features

- Multi-turn conversation using LLM
- Context window management via summarization-based memory
- Token-efficient memory design (hybrid short-term + long-term memory)
- Retrieval-Augmented Generation (RAG) for knowledge-grounded responses
- Conditional context injection without polluting conversation history
- Rule-based safety filtering
- Basic error handling

## System Design

### 1. Chat Message Schema

```json
{
    "role": "system",
    "content": "Instructions, rules, and context"
},
{  
    "role": "user",
    "content": "User input."
},
{  
    "role": "assistant",
    "content": "Model-generated response"
}
```
In this project, a conversation may contain multiple system messages.  
The first system message defines the overall instructions, rules, and context.  
The second system message stores a summary of the conversation history.

### 2. Conversation Flow

User Input  
→ Safety Check  
→ Append to Message History  
→ RAG Retrieval (optional)  
→ Prompt Construction (history + RAG context) (optional)  
→ LLM Call  
→ Append Assistant Response  
→ Memory Update  
→ Optional Summarization  

### 3. RAG Retrieval (TODO)

### 4. Memory Management
#### Hybrid Memory
- *Short-term memory*: recent messages
- *Long-term memory*: summary stored as system message

#### Optional Summarization
- Summarization triggered when context window exceeds threshold
- LLM modernized summarization

#### Optimmization:
- Avoid repeatedly summarizing existing summary to prevent information loss and over-abstraction
- Preserve recent short-term raw messages to maintain recency and accuracy
- Summarize intermediate messages to reduce context length and improve token efficiency

### 5. Fallback Degredation
#### Enable grace degradation when summary generation fails
- If a previous summary exists, preserve the system message, the existing summary, and the most recent chat history as the new context window.
- If no summary exists, preserve the system message and the most recent chat history.


### 6. Failure Handling

- API failure → rollback user message (messages.pop())


## Why This Design?

- Prevents context explosion
- Controls latency and cost
- Maintains conversation continuity

## Future Improvements

- Add RAG (retrieval-based memory): Current retrieval uses simple whitespace-based tokenization, which works for English but not for Chinese queries.
- LLM powered safety filter (optional)
- Structured output
- UI (Streamlit)