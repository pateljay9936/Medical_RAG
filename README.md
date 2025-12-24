---
title: Medical Rag Chatbot
emoji: ⚡
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---


# 🩺 Medical RAG Chatbot  
**FastAPI · LangChain · Pinecone · Google Gemini**

A **Retrieval-Augmented Generation (RAG)** based medical chatbot that provides **grounded, evidence-based medical information** using external medical documents stored in Pinecone.

The system is intentionally designed with **strict safety constraints**:
- Responses are generated **only from retrieved medical context**
- If no relevant context is found, the chatbot **refuses to guess**
- The system is for **educational purposes only**

---

## 🧠 High-Level Architecture

This project follows a **production-grade RAG architecture**, clearly separating ingestion and inference.

```mermaid
flowchart TB

    subgraph Offline_One_Time["Offline Phase (One-Time Ingestion)"]
        A[Medical Documents (PDFs)] --> B[PDF Loader]
        B --> C[Text Chunking]
        C --> D[Embedding Model (HuggingFace)]
        D --> E[Pinecone Vector Index]
    end

    subgraph Online_Runtime["Online Phase (Runtime Inference)"]
        U[User Query] --> Q[Query Classifier]
        Q -->|Medical Query| R[Pinecone Retriever]
        R --> CXT[Retrieved Context]
        CXT --> LLM[LLM (Google Gemini)]
        LLM --> S[Streaming Response (SSE)]
    end

    E --> R
```

---

## 📦 Project Phases

### Phase 1: Pinecone Index Setup (Required – One Time)

Before running ingestion or starting the chatbot, a **Pinecone index must be created manually** via the Pinecone dashboard.

**Steps:**
1. Log in to the Pinecone web dashboard
2. Create a new index
3. Configure the index with:
   - Embedding dimension matching the embedding model
   - Similarity metric (e.g. cosine)
4. Save the index name

> ⚠️ Index creation is **not handled by this codebase**.

---

### Phase 2: Data Ingestion (Mandatory – One Time)

All medical knowledge lives in Pinecone.  
The chatbot will **not work** until documents are ingested.

```bash
python ingest_pdfs.py
```

> ⚠️ If the Pinecone index is empty, the chatbot will intentionally refuse to answer.

---

### Phase 3: Vector Database (Pinecone)

- Stores document embeddings
- Performs semantic similarity search
- Acts as the **single source of truth**

No documents are stored in Docker or FastAPI.

---

### Phase 4: Backend API (FastAPI)

| Method | Endpoint | Description |
|------|--------|-------------|
| GET | `/` | Load chatbot UI |
| POST | `/get` | Stream chatbot responses |

---

### Phase 5: Query Classification

Queries are classified to decide whether retrieval is required.
This reduces hallucinations, cost, and latency.

---

### Phase 6: Retrieval-Augmented Generation (RAG)

This is **true RAG**, not prompt stuffing.

- Context is retrieved dynamically per query
- Injected automatically as `{context}`
- LLM is restricted to retrieved information only

---

### Phase 7: Streaming Responses

Responses are streamed using **Server-Sent Events (SSE)** for better UX.

---

## 🔐 Safety & Medical Constraints

- No diagnosis
- No prescriptions
- No guessing without context
- Always includes a medical disclaimer

---

## 🐳 Docker Usage (Optional)

Docker is optional and used only for runtime.
All medical knowledge remains in Pinecone.

---

## 🔹 Pinecone Namespace Strategy

The project currently operates using a **single Pinecone namespace**.

> **Although multi-namespace support is not available right now, the current design makes it easy to introduce this feature later without significant changes to the codebase.**

---

## ⚙️ Environment Variables

```bash
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
GEMINI_API_KEY=your_key
PORT=5678
```

---

## 🚀 Running the Project

```bash
python pinecone_ingession/ingest_pdfs.py         # One-time ingestion
uvicorn app:app --reload
```

Open: http://localhost:5678

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does not replace professional medical advice.
