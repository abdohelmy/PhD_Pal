# Danish PhD Course Agent

A small web app that recommends Danish PhD courses from [phdcourses.dk](https://phdcourses.dk/) based on a student's university, PhD school, research area, topic, methods, location, and ECTS preferences.

## Run

```sh
node server.mjs
```

Open `http://localhost:3000`.

This prototype has no package dependencies. `package.json` scripts are included for environments where `npm` is available.

## Python Version

If you prefer Python, run the equivalent backend with:

```sh
python3 python_agent.py
```

It serves the same frontend and implements the same main ideas: scraping, local lexical recommendations, reasoning expansion, semantic embeddings, subscriptions, and basic clustering. It also uses only the Python standard library.

To scrape once:

```sh
MAX_SCRAPE_PAGES=2 python3 python_agent.py --scrape-once
```

## What It Does

- Scrapes paginated course results from `https://phdcourses.dk/?searchWord=`.
- Visits each course detail page, extracts fuller fields, and stores a reusable course summary.
- Caches courses in `data/courses.json`.
- Filters out clearly past courses.
- Scores recommendations using research direction, interests, study programme, project topic, methods/keywords, research area, PhD school, university, location, timing, ECTS, and generic-course preference.
- Lets users choose preferred universities and allowed PhD schools as filters. These choices do not add ranking weight; they decide which courses are eligible before scoring.
- Uses semantic embeddings from either Hugging Face or a local OpenAI-compatible embedding server, then falls back to the local lexical scorer if the model call fails.
- Uses a hosted or local reasoning/chat model to expand each student profile into inferred subjects, adjacent topics, methods, and search phrases.
- Clusters saved subscribers so similar PhD profiles can be optimized together, then optionally asks the reasoning model to label clusters and suggest better search/digest strategies.
- Enriches final recommendations from individual course pages when network access is available.
- Saves email digest subscribers in `data/subscribers.json`.
- Runs a weekly digest while the server process is running.

## Commands

```sh
node server.mjs --scrape-once
node server.mjs --digest-once
```

For a shorter test scrape:

```sh
MAX_SCRAPE_PAGES=2 node server.mjs --scrape-once
```

## Email Delivery

By default, digests are written as HTML files in `data/outbox/`.

To send real email through Resend, set:

```sh
RESEND_API_KEY=...
FROM_EMAIL="Course Agent <agent@yourdomain.dk>"
```

Then run the server or `node server.mjs --digest-once`.

## Semantic Recommendations

Set a Hugging Face token to enable hosted LLM reasoning and semantic matching through Hugging Face:

```sh
HF_TOKEN=hf_...
```

Default models:

```sh
HF_EMBEDDING_MODEL=BAAI/bge-m3
HF_REASONING_MODEL=Qwen/Qwen3-14B
```

By default, the reasoning model uses Hugging Face's OpenAI-compatible chat completions router. It expands user profiles every few weeks by default, summarizes full course detail pages, and caches those results in `data/llm-insights.json`.

Embeddings are cached in `data/embeddings.json`.

To run embeddings locally too, start a second OpenAI-compatible embedding server on another port. For example:

```sh
vllm serve BAAI/bge-m3 \
  --task embed \
  --hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}' \
  --dtype auto \
  --gpu-memory-utilization 0.12 \
  --port 8001
```

Then start the app with both local endpoints:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
python3 python_agent.py
```

The local embedding endpoint does not require `HF_TOKEN`. Use `EMBEDDING_API_KEY` only if your embedding server requires a bearer token.

Before a long scrape, you can test both model endpoints with:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
python3 python_agent.py --check-models
```

The Node backend has the same model check:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
EMBEDDING_ENDPOINT=http://localhost:8001/v1/embeddings \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
HF_EMBEDDING_MODEL=BAAI/bge-m3 \
node server.mjs --check-models
```

### Local 14B Reasoning Model

With a 40 GB GPU, run the reasoning model locally behind an OpenAI-compatible server such as vLLM or SGLang. A good default for this app is `Qwen/Qwen3-14B` because it supports both thinking and non-thinking modes, has a native 32k context window, and follows structured prompts well.

Example vLLM server:

```sh
vllm serve Qwen/Qwen3-14B \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.84 \
  --port 8000
```

The model supports this larger context window. If vLLM reports an out-of-memory error while Qwen and BGE-M3 share one 40 GB GPU, reduce `--gpu-memory-utilization` for the embedding server or use a quantized Qwen model.

Then point the Python app at it:

```sh
LLM_CHAT_ENDPOINT=http://localhost:8000/v1/chat/completions \
HF_REASONING_MODEL=Qwen/Qwen3-14B \
python3 python_agent.py
```

For JSON tasks, the app defaults to asking Qwen-style models not to emit thinking text. If you want explicit thinking mode, increase the output budgets and set:

```sh
LLM_JSON_THINKING_MODE=on
PROFILE_INSIGHTS_MAX_TOKENS=2000
COURSE_SUMMARY_MAX_TOKENS=2200
LLM_RERANK_MAX_TOKENS=4000
```

To force the original local matcher:

```sh
RECOMMENDER_MODE=lexical
```

To disable LLM reasoning but keep embeddings:

```sh
REASONING_MODE=off
```

To change how often profile and cluster reasoning refreshes:

```sh
REASONING_REFRESH_DAYS=14
```

## Full Course Summaries

During scraping, the agent can visit each course URL and save a structured `courseSummary` on every course:

- `shortSummary`
- `description`
- `learningOutcomes`
- `prerequisites`
- `teachingMethods`
- `audience`
- `keywords`

Those stored summaries are used for recommendation ranking and embeddings.

## LLM Reranking After Semantic Retrieval

When embeddings are enabled, both backends use a two-stage recommendation pipeline:

1. Use embeddings to retrieve a generous candidate set from the saved course summaries.
2. Split those candidate courses into chunks of 5-10.
3. Ask the reasoning LLM to judge each chunk against the summarized user interests.
4. Merge the LLM decisions and return the final ranked recommendations.

Useful controls:

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
```

Set `LLM_RERANK_MODE=off` to use semantic similarity without the LLM reranker.

Useful controls:

```sh
SCRAPE_FULL_DETAILS=false node server.mjs --scrape-once
COURSE_DETAIL_LIMIT=20 node server.mjs --scrape-once
```

The same variables work for `python3 python_agent.py --scrape-once`.

## API

- `GET /api/metadata`
- `GET /api/courses?refresh=true`
- `POST /api/recommend`
- `POST /api/subscribe`
- `GET /api/clusters`
- `POST /api/digest/run`
- `GET /api/outbox`
