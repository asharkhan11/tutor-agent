# 📚 Tutor Agent API

> A fully free-tier, production-ready RAG AI tutor backend — Q&A on study notes, quiz generation, test papers, and concept teaching.

---

## Architecture

```
Flutter / Android / Java / Python Client
          │  HTTP (JSON + SSE)
          ▼
┌─────────────────────────────────────────────────────┐
│               FastAPI  (app/main.py)                │
│  /api/v1/documents  /qa  /quiz  /teach  /health     │
└──────────┬──────────────────────────────────────────┘
           │
  ┌────────▼─────────────────────────────────────────┐
  │            LangGraph Agent Graphs                 │
  │  QAAgent    ──► Retrieve → Grade → (Re)Answer    │
  │  QuizAgent  ──► Retrieve → Generate → Validate   │
  │  TeachAgent ──► Retrieve → Explain → Enrich      │
  └────────┬─────────────────────────────────────────┘
           │
  ┌────────▼──────────────────────┐   ┌───────────────────────┐
  │   LlamaIndex RAG              │   │  Langfuse Tracing      │
  │  SentenceSplitter (512 tok)   │   │  (Observability)       │
  │  Google gemini-embedding-001  │   └───────────────────────┘
  │  QdrantVectorStore            │
  └────────┬──────────────────────┘
           │
  ┌────────▼──────────────┐   ┌──────────────────────┐
  │   Qdrant Cloud        │   │   LLM Providers       │
  │  (Free Tier)          │   │  Groq (primary)       │
  │  per-doc collections  │   │  Cerebras (fallback)  │
  │  + global collection  │   │  llama-3.1-8b-instant │
  └───────────────────────┘   └──────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding | Google `gemini-embedding-001` via GenAI SDK | Free tier: 1,500 req/min; 3072-dim; no local download |
| LLM | Groq `llama-3.1-8b-instant` → Cerebras fallback | Fastest free tier; auto-failover on rate limits |
| Image captioning | Groq vision `meta-llama/llama-4-scout-17b-16e-instruct` | Describes diagrams/charts in PDFs before embedding |
| Vector DB | Qdrant Cloud free tier | 1 GB free, managed, no ops overhead |
| RAG orchestration | LlamaIndex | Best-in-class PDF loaders, node parsers, vector store integration |
| Agent graphs | LangGraph | Stateful multi-step reasoning with conditional edges |
| Tracing | Langfuse | Open-source, free hosted tier, full LLM observability |
| Chunk size | 512 tokens / 64 overlap | Balances context quality vs. retrieval precision |

### Upload Flow (corrected)

```
POST /documents/upload
        │
        ▼ (202 Accepted immediately — doc_id returned)
FastAPI BackgroundTask
        │
        ├─► Save PDF to disk
        ├─► Extract text per page (pdfplumber)
        ├─► Extract images per page (pymupdf)
        │      └─► Describe images via Groq vision  ← Groq is called HERE only
        ├─► Chunk text (LlamaIndex SentenceSplitter)
        ├─► Embed chunks (gemini-embedding-001 via Google GenAI SDK) ← embedding service
        └─► Upsert into Qdrant (per-doc + global collection)

Poll GET /documents/{doc_id}  →  status: processing | ready | failed
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone <your-repo>
cd tutor-agent
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Run locally (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

### 3. Run with Docker

```bash
docker-compose up --build
```

### 4. Open API docs

```
http://localhost:8000/docs
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | From [console.groq.com](https://console.groq.com) — used for **LLM** and **image captioning** |
| `CEREBRAS_API_KEY` | ✅ | — | From [cloud.cerebras.ai](https://cloud.cerebras.ai) — LLM fallback |
| `GEMINI_API_KEY` | ✅ | — | From [aistudio.google.com](https://aistudio.google.com/app/apikey) — used for **embeddings only** |
| `QDRANT_URL` | ✅ | — | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | — | From Qdrant Cloud dashboard |
| `LANGFUSE_PUBLIC_KEY` | ☑️ | — | From [cloud.langfuse.com](https://cloud.langfuse.com) (optional but recommended) |
| `LANGFUSE_SECRET_KEY` | ☑️ | — | Langfuse secret |
| `PRIMARY_LLM_PROVIDER` | — | `groq` | `groq` or `cerebras` |
| `GROQ_MODEL` | — | `llama-3.1-8b-instant` | Groq chat model |
| `GROQ_VISION_MODEL` | — | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision model for image captioning |
| `EMBED_MODEL` | — | `gemini-embedding-001` | Gemini embedding model (`text-embedding-004` deprecated Nov 2025) |
| `EMBED_DIMENSION` | — | `3072` | Must match the embedding model output (`gemini-embedding-001` is 3072-dim) |
| `CHUNK_SIZE` | — | `512` | Token chunk size for SentenceSplitter |
| `CHUNK_OVERLAP` | — | `64` | Overlap between adjacent chunks |
| `MAX_UPLOAD_SIZE_MB` | — | `50` | PDF size limit |
| `QDRANT_COLLECTION_NAME` | — | `tutor_notes` | Name of the global Qdrant collection |

> **Note on API key responsibilities:**
> - `GEMINI_API_KEY` → embedding only (called during document upload and at query time)
> - `GROQ_API_KEY` → LLM chat/completions (Q&A, quiz, teach agents) + vision image captioning
> - The two are completely separate — Groq is never called for embeddings

---

## API Reference

### Documents

#### Upload a PDF
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file=<pdf_file>
subject=Physics          (optional)
chapter=Chapter 3        (optional)
tags=thermodynamics,heat (optional, comma-separated)
```
**Response (202 Accepted — immediately):**
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Upload received. Ingestion running — poll /documents/{doc_id} for status.",
  "status": "processing"
}
```
Poll `GET /api/v1/documents/{doc_id}` until `status` is `ready` or `failed`.

#### List documents
```http
GET /api/v1/documents
```

#### Get document status
```http
GET /api/v1/documents/{doc_id}
```
```json
{
  "doc_id": "550e8400-...",
  "filename": "physics_notes.pdf",
  "subject": "Physics",
  "chapter": "Chapter 3",
  "tags": ["thermodynamics", "heat"],
  "page_count": 24,
  "chunk_count": 52,
  "has_images": true,
  "status": "ready",
  "error": null
}
```

#### Delete document
```http
DELETE /api/v1/documents/{doc_id}
```

---

### Q&A

#### Ask a question
```http
POST /api/v1/qa
Content-Type: application/json

{
  "question": "Explain the first law of thermodynamics",
  "doc_ids": ["550e8400-..."],   // omit to search ALL docs
  "top_k": 5,
  "session_id": "user-session-abc",
  "user_id": "student-001"
}
```
**Response:**
```json
{
  "answer": "The first law of thermodynamics states...",
  "sources": [
    {
      "doc_id": "550e8400-...",
      "filename": "physics_notes.pdf",
      "page": 12,
      "score": 0.923,
      "text": "...chunk preview..."
    }
  ],
  "session_id": "user-session-abc",
  "tokens_used": 312
}
```

#### Streaming Q&A (SSE)
```http
POST /api/v1/qa/stream
Content-Type: application/json
Accept: text/event-stream

{ "question": "...", "doc_ids": [...] }
```
Each event: `data: {"delta": "word ", "done": false}`  
Final event: `data: {"done": true, "answer": "...", "sources": [...]}`

---

### Quiz Generation

#### Generate a quiz
```http
POST /api/v1/quiz/generate
Content-Type: application/json

{
  "doc_ids": ["550e8400-..."],
  "topic": "Newton's Laws",          // optional — focuses retrieval
  "chapter": "Chapter 5",            // optional
  "num_questions": 10,
  "question_types": ["mcq", "short", "true_false"],
  "difficulty": "mixed",             // easy | medium | hard | mixed
  "session_id": "quiz-session-1"
}
```
**Response:**
```json
{
  "quiz_id": "...",
  "title": "Quiz: Newton's Laws",
  "questions": [
    {
      "id": "...",
      "type": "mcq",
      "question": "Which law states F = ma?",
      "options": [
        {"key": "A", "text": "Newton's First Law"},
        {"key": "B", "text": "Newton's Second Law"},
        {"key": "C", "text": "Newton's Third Law"},
        {"key": "D", "text": "Law of Gravitation"}
      ],
      "answer": "B",
      "explanation": "Newton's Second Law relates force, mass and acceleration...",
      "difficulty": "easy"
    }
  ],
  "session_id": "quiz-session-1"
}
```

#### Generate a test paper
```http
POST /api/v1/quiz/test-paper
Content-Type: application/json

{
  "doc_ids": ["550e8400-..."],
  "title": "Mid-Term Physics Test",
  "chapters": ["Chapter 3", "Chapter 4"],
  "total_marks": 100,
  "duration_minutes": 90,
  "difficulty": "medium",
  "sections": [
    {"type": "mcq",   "count": 20, "marks_each": 1},
    {"type": "short", "count": 5,  "marks_each": 6},
    {"type": "long",  "count": 2,  "marks_each": 15}
  ]
}
```

---

### Teaching

#### Explain a concept
```http
POST /api/v1/teach
Content-Type: application/json

{
  "concept": "Entropy",
  "doc_ids": ["550e8400-..."],
  "detail_level": "detailed",     // brief | medium | detailed
  "include_examples": true,
  "include_analogies": true,
  "session_id": "teach-session-1"
}
```
**Response:**
```json
{
  "concept": "Entropy",
  "explanation": "Entropy is a measure of disorder...",
  "key_points": ["Entropy always increases in isolated systems", "..."],
  "examples": ["Ice melting in water increases entropy...", "..."],
  "analogies": ["Think of entropy like a teenager's bedroom..."],
  "related_concepts": ["Thermodynamics", "Gibbs Free Energy", "..."],
  "sources": [...],
  "session_id": "teach-session-1"
}
```

---

### Health

```http
GET /api/v1/health
```
```json
{
  "status": "ok",
  "qdrant": "ok",
  "llm_provider": "groq",
  "embed_model": "gemini-embedding-001",
  "langfuse": "enabled"
}
```

---

## Flutter / Android Integration

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

const BASE = 'http://your-server:8000/api/v1';

// Upload a PDF (returns immediately with doc_id to poll)
Future<String> uploadPdf(File pdf, {String? subject, String? chapter}) async {
  final req = http.MultipartRequest('POST', Uri.parse('$BASE/documents/upload'));
  req.files.add(await http.MultipartFile.fromPath('file', pdf.path));
  if (subject != null) req.fields['subject'] = subject;
  if (chapter != null) req.fields['chapter'] = chapter;
  final res = await req.send();
  final body = jsonDecode(await res.stream.bytesToString());
  return body['doc_id'];  // status will be 'processing'; poll GET /documents/{doc_id}
}

// Poll until ready
Future<bool> waitForIngestion(String docId) async {
  for (int i = 0; i < 30; i++) {
    await Future.delayed(const Duration(seconds: 3));
    final res = await http.get(Uri.parse('$BASE/documents/$docId'));
    final body = jsonDecode(res.body);
    if (body['status'] == 'ready') return true;
    if (body['status'] == 'failed') throw Exception(body['error']);
  }
  throw TimeoutException('Ingestion took too long');
}

// Ask a question
Future<Map> askQuestion(String question, List<String> docIds) async {
  final res = await http.post(
    Uri.parse('$BASE/qa'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'question': question, 'doc_ids': docIds}),
  );
  return jsonDecode(res.body);
}

// Generate a quiz
Future<Map> generateQuiz(List<String> docIds, {String? topic, int numQ = 10}) async {
  final res = await http.post(
    Uri.parse('$BASE/quiz/generate'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'doc_ids': docIds,
      'topic': topic,
      'num_questions': numQ,
      'question_types': ['mcq', 'short'],
      'difficulty': 'mixed',
    }),
  );
  return jsonDecode(res.body);
}

// Streaming Q&A — SSE
Stream<String> streamAnswer(String question, List<String> docIds) async* {
  final client = http.Client();
  final req = http.Request('POST', Uri.parse('$BASE/qa/stream'));
  req.headers['Content-Type'] = 'application/json';
  req.body = jsonEncode({'question': question, 'doc_ids': docIds});
  final response = await client.send(req);
  await for (final chunk in response.stream.transform(utf8.decoder)) {
    for (final line in chunk.split('\n')) {
      if (line.startsWith('data: ')) {
        final data = jsonDecode(line.substring(6));
        if (data['done'] == false) yield data['delta'] as String;
      }
    }
  }
}
```

## Java / Spring Boot Integration

```java
// Using Spring's WebClient
@Service
public class TutorAgentClient {
    private final WebClient client = WebClient.builder()
        .baseUrl("http://your-server:8000/api/v1")
        .build();

    public Mono<Map> askQuestion(String question, List<String> docIds) {
        return client.post().uri("/qa")
            .bodyValue(Map.of("question", question, "doc_ids", docIds))
            .retrieve()
            .bodyToMono(Map.class);
    }

    public Mono<Map> generateQuiz(List<String> docIds, String topic, int numQ) {
        return client.post().uri("/quiz/generate")
            .bodyValue(Map.of(
                "doc_ids", docIds,
                "topic", topic,
                "num_questions", numQ,
                "question_types", List.of("mcq", "short"),
                "difficulty", "mixed"
            ))
            .retrieve()
            .bodyToMono(Map.class);
    }
}
```

---

## Testing

```bash
# Run all tests (requires .env with valid credentials)
pytest tests/ -v --asyncio-mode=auto

# Run only health check (no external APIs needed)
pytest tests/ -v -k "health"
```

---

## Free-Tier Resource Usage

| Resource | Free Tier Limit | This App's Usage |
|----------|-----------------|------------------|
| Groq API | 14,400 req/day, 30 RPM | ~2–5 req per user interaction (LLM) + 1 req per image during upload |
| Cerebras | 60 RPM | Fallback only |
| Gemini Embedding API (`gemini-embedding-001`) | 1,500 req/min | 1 req per chunk during upload; 1 req per query at retrieval time |
| Qdrant Cloud | 1 GB RAM | ~1,000 pages per 100 MB; one 50-page book ≈ ~200 chunks ≈ ~20 MB |
| Langfuse Cloud | 50K events/month | 1 event per agent invocation |

### Staying in free tier
- Groq's `llama-3.1-8b-instant` averages ~0.5 s per call — fast and cheap
- Chunk size 512 ≈ 1 page per 2 chunks; a 100-page textbook ≈ 400 chunks ≈ 400 Gemini embed calls on first upload
- Subsequent queries only use 1 Gemini embed call (the query itself)
- Set `CHUNK_SIZE=256` for denser, smaller vectors if Qdrant storage is tight

---

## Extending the System

### Add a new agent (e.g. Flashcard Generator)
1. Create `app/agents/flashcard_agent.py` following the same LangGraph pattern
2. Add request/response schemas to `app/models/schemas.py`
3. Create `app/api/routes/flashcard.py`
4. Register in `app/api/__init__.py`

### Use a different embedding model
Change `EMBED_MODEL` and `EMBED_DIMENSION` in `.env`. The app uses the Gemini embedding API — supported models include:
- `gemini-embedding-001` (default, 3072-dim)
- `text-embedding-004` (768-dim, **deprecated** Nov 2025 — do not use for new projects)

If you switch to a local FastEmbed model instead, replace `app/core/embeddings.py` with the FastEmbed equivalent and update `requirements.txt` accordingly.

> **⚠️ Dimension warning:** If you re-ingest documents after changing `EMBED_MODEL`, you **must** delete and recreate your Qdrant collections. Vectors from `text-embedding-004` (768-dim) and `gemini-embedding-001` (3072-dim) are incompatible — mixing them will cause silent retrieval failures.

### Switch to a different vector DB
Replace `app/core/vector_store.py` and the `QdrantVectorStore` calls in `ingestor.py` and `retriever.py` with any LlamaIndex-supported vector store (Pinecone, Weaviate, Chroma, etc.).
