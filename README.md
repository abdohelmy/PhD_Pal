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

It serves the same frontend and implements the same main ideas: scraping, local lexical recommendations, Hugging Face reasoning expansion, Hugging Face embeddings, subscriptions, and basic clustering. It also uses only the Python standard library.

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
- Uses Hugging Face semantic embeddings when `HF_TOKEN` is set, then falls back to the local lexical scorer if the model call fails.
- Uses a Hugging Face reasoning/chat model when `HF_TOKEN` is set to expand each student profile into inferred subjects, adjacent topics, methods, and search phrases.
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

## Hugging Face Semantic Recommendations

Set a Hugging Face token to enable LLM reasoning and semantic matching:

```sh
HF_TOKEN=hf_...
```

Default models:

```sh
HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
HF_REASONING_MODEL=deepseek-ai/DeepSeek-R1:fastest
```

The reasoning model uses Hugging Face's OpenAI-compatible chat completions router. It expands user profiles every few weeks by default, summarizes full course detail pages, and caches those results in `data/llm-insights.json`.

Embeddings are cached in `data/embeddings.json`.

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

When `HF_TOKEN` is set, the Python backend uses a two-stage recommendation pipeline:

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
