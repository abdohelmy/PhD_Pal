# Danish PhD Course Agent

A small web app that recommends Danish PhD courses from [phdcourses.dk](https://phdcourses.dk/) based on a student's university, PhD school, research area, research direction, interests, methods, location, and ECTS preferences.

The project has two backend options:

- Python backend: `python_agent.py`, configured through `python_config.py`.
- Node/JS backend: `server.mjs`, configured through environment variables in the shell.

Both backends serve the same frontend from `public/`.

## What It Does

- Scrapes paginated course results from `https://phdcourses.dk/?searchWord=`.
- Visits each course detail page, extracts fuller fields, and stores a reusable course summary.
- Uses a reasoning LLM to summarize courses and expand student research interests.
- Uses embeddings to retrieve a broad candidate set from course summaries.
- Uses the reasoning LLM again to rerank candidate chunks into final recommendations.
- Lets users choose preferred universities and allowed PhD schools as filters, without adding ranking weight.
- Clusters saved subscribers and can ask the reasoning model to label clusters and suggest better search/digest strategies.
- Saves email digest subscribers in `data/subscribers.json`.
- Writes digest emails to `data/outbox/` unless a Resend API key is configured.

## Python Backend

Start here if you prefer Python:

```sh
python3 python_agent.py
```

Open:

```text
http://localhost:3000
```

The Python backend uses only the Python standard library. Its configuration lives in:

```text
python_config.py
```

Change defaults there, or override them from the command line with environment variables.

### Python Commands

Run the web app:

```sh
python3 python_agent.py
```

Scrape courses once:

```sh
MAX_SCRAPE_PAGES=2 python3 python_agent.py --scrape-once
```

Check local or hosted model endpoints:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
python3 python_agent.py --check-models
```

### Python Config

The main settings are in `python_config.py`:

```py
HF_EMBEDDING_MODEL = "BAAI/bge-m3"
HF_REASONING_MODEL = "Qwen/Qwen3-14B"
SEMANTIC_CANDIDATE_LIMIT = 60
LLM_RERANK_CHUNK_SIZE = 8
COURSE_SUMMARY_MAX_TOKENS = 1400
LLM_RERANK_MAX_TOKENS = 2200
```

The file reads environment variables too, so this still works:

```sh
REASONING_MODE=off python3 python_agent.py
```

Useful Python environment overrides:

```sh
PORT=3000
MAX_SCRAPE_PAGES=120
SCRAPE_FULL_DETAILS=true
COURSE_DETAIL_LIMIT=20
RECOMMENDER_MODE=semantic
REASONING_MODE=on
LLM_RERANK_MODE=on
LLM_RERANK_FINAL_PASS=true
REASONING_REFRESH_DAYS=21
```

## Local Models

Use these model servers for either backend.

Start Qwen3-14B for reasoning:

One-line version:

```sh
vllm serve Qwen/Qwen3-14B --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.84 --port 8000
```

Multi-line version:

```sh
vllm serve Qwen/Qwen3-14B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.84 \
  --port 8000
```

Start BGE-M3 for embeddings:

One-line version:

```sh
vllm serve BAAI/bge-m3 --hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}' --dtype auto --gpu-memory-utilization 0.12 --port 8001
```

Multi-line version:

```sh
vllm serve BAAI/bge-m3 \
  --hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}' \
  --dtype auto \
  --gpu-memory-utilization 0.12 \
  --port 8001
```

If you use the multi-line version, the `\` must be the final character on the line. Do not put spaces after it. Some vLLM versions support `--task embed`; yours does not, so leave it out.

Then run the Python backend with:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
python3 python_agent.py
```

Local endpoints do not require `HF_TOKEN`. Use `LLM_API_KEY` or `EMBEDDING_API_KEY` only if your local server requires bearer auth.

## Hosted Hugging Face Models

If you do not run models locally, set:

```sh
HF_TOKEN=hf_...
python3 python_agent.py
```

By default, hosted reasoning uses Hugging Face's OpenAI-compatible chat completions router. Hosted embeddings use Hugging Face feature extraction.

## Node/JS Backend

Run the JavaScript backend with:

```sh
node server.mjs
```

Open:

```text
http://localhost:3000
```

The Node backend uses the same environment variables as the Python backend, but it does not use `python_config.py`.

### Node Commands

Run the web app:

```sh
node server.mjs
```

Scrape courses once:

```sh
MAX_SCRAPE_PAGES=2 node server.mjs --scrape-once
```

Run weekly digest once:

```sh
node server.mjs --digest-once
```

Check model endpoints:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
node server.mjs --check-models
```

Run Node with local models:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
node server.mjs
```

## Recommendation Pipeline

1. Scrape upcoming courses from `phdcourses.dk`.
2. Visit each course link and extract full details.
3. Ask the reasoning LLM to summarize each course into structured fields.
4. Embed the user profile and course summaries.
5. Retrieve a generous semantic candidate set.
6. Split candidates into chunks of 5-10 courses.
7. Ask the reasoning LLM to decide `recommend`, `maybe`, or `exclude`.
8. Return the final ranked recommendations.

Useful controls for both backends:

```sh
SEMANTIC_CANDIDATE_LIMIT=60
SEMANTIC_CANDIDATE_THRESHOLD=0.25
LLM_RERANK_CHUNK_SIZE=8
LLM_RERANK_FINAL_PASS=true
LLM_RERANK_MODE=on
PROFILE_INSIGHTS_MAX_TOKENS=1200
COURSE_SUMMARY_MAX_TOKENS=1400
LLM_RERANK_MAX_TOKENS=2200
CLUSTER_INSIGHTS_MAX_TOKENS=1800
LLM_JSON_THINKING_MODE=off
```

Set `LLM_RERANK_MODE=off` to use semantic similarity without the LLM reranker.

Set `RECOMMENDER_MODE=lexical` to use the original local keyword matcher.

## Email Delivery

By default, digests are written as HTML files in `data/outbox/`.

To send real email through Resend, set:

```sh
RESEND_API_KEY=...
FROM_EMAIL="Course Agent <agent@yourdomain.dk>"
```

## API

- `GET /api/metadata`
- `GET /api/courses?refresh=true`
- `POST /api/recommend`
- `POST /api/subscribe`
- `GET /api/clusters`
- `POST /api/digest/run`
- `GET /api/outbox`
