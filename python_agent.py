#!/usr/bin/env python3
"""
Python version of the Danish PhD course recommendation agent.

Run:
    python3 python_agent.py

Optional Hugging Face support:
    HF_TOKEN=hf_... python3 python_agent.py

The code intentionally uses only the Python standard library so it is easier to
read, copy, and modify while learning.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
DATA_DIR = ROOT / "data"
COURSE_CACHE = DATA_DIR / "courses.json"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"
EMBEDDING_CACHE = DATA_DIR / "embeddings.json"
LLM_INSIGHTS_CACHE = DATA_DIR / "llm-insights.json"
OUTBOX_DIR = DATA_DIR / "outbox"

SOURCE_BASE = "https://phdcourses.dk"
USER_AGENT = "Danish PhD Course Recommendation Agent Python/0.1"
PORT = int(os.getenv("PORT", "3000"))
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
HF_REASONING_MODEL = os.getenv("HF_REASONING_MODEL", "deepseek-ai/DeepSeek-R1:fastest")
HF_FEATURE_ENDPOINT = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"
)
HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
REASONING_REFRESH_DAYS = int(os.getenv("REASONING_REFRESH_DAYS", "21"))


FIELD_KEYWORDS = {
    "AI, data science, statistics": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "data science",
        "python",
        "statistics",
        "causal",
        "stochastic",
        "large language model",
    ],
    "Health, medicine, life science": [
        "health",
        "medical",
        "medicine",
        "clinical",
        "epidemiology",
        "biostatistics",
        "pharmaceutical",
        "biology",
        "patient",
    ],
    "Engineering, energy, environment": [
        "engineering",
        "energy",
        "environment",
        "sustainability",
        "climate",
        "hydrogen",
        "wind",
        "transport",
        "materials",
    ],
    "Humanities, design, arts": [
        "humanities",
        "arts",
        "history",
        "culture",
        "design",
        "architecture",
        "language",
        "communication",
    ],
    "Social science, business, law": [
        "social",
        "business",
        "management",
        "economics",
        "law",
        "political",
        "qualitative",
        "policy",
    ],
    "Generic research skills": [
        "ethics",
        "research integrity",
        "teaching",
        "project management",
        "academic writing",
        "literature search",
        "career",
    ],
}

UNIVERSITIES = [
    "Copenhagen Business School",
    "Design school Kolding",
    "IT University of Copenhagen",
    "Roskilde University",
    "Technical University of Denmark",
    "The Royal Danish Academy",
    "University of Copenhagen",
    "University of Greenland",
    "University of Southern Denmark",
    "Aalborg University",
    "Aarhus School of Architecture",
    "Aarhus University",
]

STOP_WORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "how",
    "into",
    "not",
    "the",
    "this",
    "that",
    "with",
    "your",
    "their",
    "using",
    "within",
    "about",
    "based",
    "study",
    "course",
    "courses",
    "goals",
    "learning",
    "phd",
    "research",
    "support",
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTBOX_DIR.mkdir(exist_ok=True)


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fallback


def write_json(path: Path, data: Any) -> None:
    ensure_dirs()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]*>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9æøå]+", " ", value.lower()).strip()


def summarize(value: str, max_len: int = 280) -> str:
    text = clean_text(value)
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    return cut[: max(cut.rfind(" "), 140)].strip() + "..."


def fetch_text(url: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> str:
    data = None
    headers = {"user-agent": USER_AGENT, "accept": "text/html,application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["content-type"] = "application/json"
    request = Request(url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def post_json(url: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    request_headers = {"content-type": "application/json", **(headers or {})}
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers=request_headers,
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(str(error)) from error


def infer_university(phd_school: str) -> str:
    lower = phd_school.lower()
    if "aarhus" in lower:
        return "Aarhus University"
    if "aalborg" in lower:
        return "Aalborg University"
    if "southern denmark" in lower or "sdu" in lower:
        return "University of Southern Denmark"
    if "copenhagen" in lower or "ucph" in lower:
        if "business" in lower or "cbs" in lower:
            return "Copenhagen Business School"
        return "University of Copenhagen"
    if "dtu" in lower:
        return "Technical University of Denmark"
    if "roskilde" in lower:
        return "Roskilde University"
    if "greenland" in lower:
        return "University of Greenland"
    if "royal danish academy" in lower:
        return "The Royal Danish Academy"
    if "design school kolding" in lower:
        return "Design school Kolding"
    if "it university" in lower:
        return "IT University of Copenhagen"
    return ""


def parse_ects(value: str) -> float | None:
    match = re.search(r"(\d+(?:[,.]\d+)?)", html.unescape(value))
    return float(match.group(1).replace(",", ".")) if match else None


def parse_total_pages(page_html: str) -> int:
    match = re.search(r'<article class="pagination">[\s\S]*?of\s+(\d+)', page_html, re.I)
    return int(match.group(1)) if match else 1


def parse_course_rows(page_html: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r'<tbody>\s*<tr>\s*<td[^>]*>\s*<a class="subtitle" href="([^"]+)">([\s\S]*?)</a>\s*</td>\s*'
        r"<td>\s*([\s\S]*?)</td>\s*<td>\s*([\s\S]*?)</td>\s*<td>\s*([\s\S]*?)</td>\s*</tr>\s*"
        r'<tr>\s*<td colspan="4">\s*<a href="/phdSchool/(\d+)"[\s\S]*?>([\s\S]*?)</a>\s*'
        r"<p[^>]*>([\s\S]*?)</p>",
        re.I,
    )
    courses = []
    for match in pattern.finditer(page_html):
        href, title, city, start_date, ects, school_id, phd_school, description = match.groups()
        course_id = re.search(r"/Course/(\d+)", href)
        phd_school_text = clean_text(phd_school)
        courses.append(
            {
                "id": course_id.group(1) if course_id else href,
                "title": clean_text(title),
                "sourceUrl": urljoin(SOURCE_BASE, href),
                "city": clean_text(city),
                "startDate": clean_text(start_date),
                "ects": clean_text(ects),
                "ectsValue": parse_ects(ects),
                "phdSchoolId": school_id,
                "phdSchool": phd_school_text,
                "university": infer_university(phd_school_text),
                "description": summarize(description, 520),
                "scrapedAt": now_iso(),
            }
        )
    return courses


def text_after_heading(page_html: str, heading: str) -> str:
    pattern = re.compile(
        rf"<h3[^>]*>\s*{re.escape(heading)}\s*</h3>\s*([\s\S]*?)(?=<h3|<h2|</main|<div id=\"show-cookie)",
        re.I,
    )
    match = pattern.search(page_html)
    return clean_text(match.group(1)) if match else ""


def parse_course_detail(page_html: str, fallback: dict[str, Any]) -> dict[str, Any]:
    title_match = re.search(r"<h2[^>]*>\s*([\s\S]*?)\s*</h2>", page_html, re.I)
    school_match = re.search(r"<h3[^>]*>\s*([\s\S]*?)\s*</h3>", page_html, re.I)
    description_match = re.search(
        r"<h3[^>]*>[\s\S]*?</h3>\s*([\s\S]*?)\s*<a[^>]*>\s*Back\s*</a>",
        page_html,
        re.I,
    )
    registration_match = re.search(
        r"<h3[^>]*>\s*Link\s*</h3>\s*<a[^>]*href=\"([^\"]+)\"",
        page_html,
        re.I,
    )
    title = clean_text(title_match.group(1)) if title_match else fallback.get("title", "")
    phd_school = clean_text(school_match.group(1)) if school_match else fallback.get("phdSchool", "")
    full_description = clean_text(description_match.group(1)) if description_match else fallback.get("description", "")

    detail = {
        **fallback,
        "title": title,
        "phdSchool": phd_school,
        "university": infer_university(phd_school),
        "description": summarize(full_description, 620),
        "fullDescription": full_description,
        "startDate": text_after_heading(page_html, "Course dates") or fallback.get("startDate", ""),
        "lecturer": text_after_heading(page_html, "Lecturer"),
        "venue": text_after_heading(page_html, "Place/Venue"),
        "city": text_after_heading(page_html, "City") or fallback.get("city", ""),
        "ects": text_after_heading(page_html, "ECTS") or fallback.get("ects", ""),
        "registrationUrl": html.unescape(registration_match.group(1)) if registration_match else "",
    }
    detail["rawCourseText"] = clean_text(
        " ".join(
            [
                title,
                phd_school,
                full_description,
                detail.get("startDate", ""),
                detail.get("lecturer", ""),
                detail.get("venue", ""),
                detail.get("city", ""),
                detail.get("ects", ""),
            ]
        )
    )
    return detail


def fetch_course_detail(course: dict[str, Any]) -> dict[str, Any]:
    try:
        return parse_course_detail(fetch_text(course["sourceUrl"]), course)
    except Exception:
        return course


def fallback_course_summary(course: dict[str, Any]) -> dict[str, Any]:
    text = course.get("fullDescription") or course.get("description", "")
    keywords = list(
        dict.fromkeys(
            [
                *extract_terms(course.get("title")),
                *extract_terms(course.get("phdSchool")),
                *extract_terms(text),
            ]
        )
    )[:12]
    return {
        "mode": "extractive",
        "model": None,
        "generatedAt": now_iso(),
        "shortSummary": summarize(text, 360),
        "description": summarize(text, 700),
        "learningOutcomes": [],
        "prerequisites": "",
        "teachingMethods": "",
        "audience": "",
        "keywords": keywords,
    }


def course_detail_text(course: dict[str, Any]) -> str:
    fields = [
        ("Title", course.get("title")),
        ("PhD school", course.get("phdSchool")),
        ("University", course.get("university")),
        ("ECTS", course.get("ects")),
        ("Dates", course.get("startDate")),
        ("City", course.get("city")),
        ("Lecturer", course.get("lecturer")),
        ("Venue", course.get("venue")),
        ("Full course text", course.get("rawCourseText") or course.get("fullDescription") or course.get("description")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def summarize_course_with_llm(course: dict[str, Any]) -> dict[str, Any]:
    text = course_detail_text(course)
    cache = read_json(LLM_INSIGHTS_CACHE, {"entries": {}})
    key = f"course-summary:{HF_REASONING_MODEL}:{sha_key(text)}"
    cached = cache["entries"].get(key)
    if cached:
        return cached["summary"]

    if not reasoning_enabled():
        summary = fallback_course_summary(course)
        cache["entries"][key] = {"generatedAt": now_iso(), "summary": summary}
        write_json(LLM_INSIGHTS_CACHE, cache)
        return summary

    try:
        content = call_reasoning_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You summarize PhD course pages for a course recommendation system. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Course page text:\n{text[:9000]}\n\n"
                        "Return JSON with keys: shortSummary, description, learningOutcomes, "
                        "prerequisites, teachingMethods, audience, keywords. Focus on what a PhD "
                        "student needs to decide whether the course is relevant."
                    ),
                },
            ],
            max_tokens=1100,
        )
        raw = parse_model_json(content)
        summary = {
            "mode": "llm",
            "model": HF_REASONING_MODEL,
            "generatedAt": now_iso(),
            "shortSummary": summarize(str(raw.get("shortSummary") or raw.get("summary") or course.get("description", "")), 360),
            "description": summarize(str(raw.get("description") or course.get("fullDescription") or course.get("description", "")), 900),
            "learningOutcomes": clean_list(raw.get("learningOutcomes") or raw.get("outcomes"), 8),
            "prerequisites": summarize(str(raw.get("prerequisites", "")), 240),
            "teachingMethods": summarize(str(raw.get("teachingMethods") or raw.get("methods") or ""), 240),
            "audience": summarize(str(raw.get("audience", "")), 220),
            "keywords": clean_list(raw.get("keywords"), 14),
        }
        cache["entries"][key] = {"generatedAt": now_iso(), "summary": summary}
        write_json(LLM_INSIGHTS_CACHE, cache)
        return summary
    except Exception:
        return fallback_course_summary(course)


def enrich_and_summarize_course(course: dict[str, Any]) -> dict[str, Any]:
    detail = fetch_course_detail(course)
    course_summary = summarize_course_with_llm(detail)
    return {
        **detail,
        "courseSummary": course_summary,
        "description": course_summary.get("shortSummary") or detail.get("description", ""),
    }


def is_upcoming(course: dict[str, Any]) -> bool:
    date_text = course.get("startDate", "")
    match = re.search(r"\b(20\d{2})\b", date_text)
    if not match:
        return True
    year = int(match.group(1))
    current_year = datetime.now().year
    if year > current_year:
        return True
    if year < current_year:
        return False
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    lower = date_text.lower()
    month_index = next((i for i, month in enumerate(months) if month in lower), None)
    return month_index is None or month_index >= datetime.now().month - 1


def scrape_courses(max_pages: int | None = None) -> dict[str, Any]:
    first_html = fetch_text(f"{SOURCE_BASE}/?searchWord=")
    total_pages = parse_total_pages(first_html)
    if max_pages:
        total_pages = min(total_pages, max_pages)

    course_map = {course["id"]: course for course in parse_course_rows(first_html)}
    for page in range(2, total_pages + 1):
        page_html = fetch_text(f"{SOURCE_BASE}/?page={page}&currentSearchWord=&")
        for course in parse_course_rows(page_html):
            course_map[course["id"]] = course

    listed_courses = sorted(course_map.values(), key=lambda c: f"{c.get('startDate')} {c.get('title')}")
    summarize_details = os.getenv("SCRAPE_FULL_DETAILS", "true") != "false"
    detail_limit = int(os.getenv("COURSE_DETAIL_LIMIT", str(len(listed_courses))))
    courses = []
    for course in listed_courses:
        if summarize_details and len(courses) < detail_limit:
            courses.append(enrich_and_summarize_course(course))
        else:
            courses.append({**course, "courseSummary": fallback_course_summary(course)})

    cache = {
        "source": SOURCE_BASE,
        "scrapedAt": now_iso(),
        "totalPages": total_pages,
        "count": len(courses),
        "courses": courses,
    }
    write_json(COURSE_CACHE, cache)
    return cache


def get_course_cache() -> dict[str, Any]:
    cache = read_json(COURSE_CACHE, None)
    if cache:
        return cache
    return {"source": SOURCE_BASE, "scrapedAt": None, "count": 0, "courses": []}


def unique_sorted(values: list[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def get_phd_schools(courses: list[dict[str, Any]]) -> list[str]:
    return unique_sorted([course.get("phdSchool") for course in courses])


def selected_universities(profile: dict[str, Any]) -> list[str]:
    values = list(profile.get("preferredUniversities") or [])
    if profile.get("university"):
        values.append(profile["university"])
    return unique_sorted(values)


def matches_own_school(profile: dict[str, Any], course: dict[str, Any]) -> bool:
    return bool(
        profile.get("school")
        and profile["school"].lower() in course.get("phdSchool", "").lower()
    )


def course_allowed_by_profile(profile: dict[str, Any], course: dict[str, Any]) -> bool:
    universities = selected_universities(profile)
    if universities and course.get("university") not in universities:
        return False

    allowed_other_schools = profile.get("allowedOtherSchools") or []
    if not profile.get("includeOtherSchools") and profile.get("school"):
        return matches_own_school(profile, course)
    if profile.get("includeOtherSchools") and allowed_other_schools:
        return matches_own_school(profile, course) or course.get("phdSchool") in allowed_other_schools
    return True


def hf_token() -> str:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""


def reasoning_enabled() -> bool:
    return bool(hf_token()) and os.getenv("REASONING_MODE") != "off"


def embedding_enabled() -> bool:
    return bool(hf_token()) and os.getenv("RECOMMENDER_MODE") != "lexical"


def sha_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def profile_text(profile: dict[str, Any]) -> str:
    fields = [
        ("Research area", profile.get("area")),
        ("Study programme", profile.get("studyProgram")),
        ("PhD school", profile.get("school")),
        ("Research direction", profile.get("researchDirection")),
        ("Interests and learning goals", profile.get("interests")),
        ("Project topic", profile.get("topic")),
        ("Methods and keywords", profile.get("keywords") or profile.get("methods")),
        ("LLM inferred subjects", profile.get("llmSubjects")),
        ("LLM adjacent subjects", profile.get("llmAdjacentSubjects")),
        ("LLM search phrases", profile.get("llmSearchKeywords")),
        ("LLM suggested methods", profile.get("llmMethods")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def course_text(course: dict[str, Any]) -> str:
    summary = course.get("courseSummary") or {}
    fields = [
        ("Course", course.get("title")),
        ("PhD school", course.get("phdSchool")),
        ("University", course.get("university")),
        ("Curated summary", summary.get("shortSummary")),
        ("Description", summary.get("description") or course.get("description")),
        ("Learning outcomes", "; ".join(summary.get("learningOutcomes") or [])),
        ("Prerequisites", summary.get("prerequisites")),
        ("Teaching methods", summary.get("teachingMethods")),
        ("Audience", summary.get("audience")),
        ("Keywords", ", ".join(summary.get("keywords") or [])),
        ("ECTS", course.get("ects")),
        ("Dates", course.get("startDate")),
        ("City", course.get("city")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def parse_model_json(content: str) -> dict[str, Any]:
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.I).strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, re.I)
    if fenced:
        content = fenced.group(1)
    else:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            content = match.group(0)
    return json.loads(content)


def clean_list(value: Any, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for item in value:
        text = clean_text(str(item))
        if 2 < len(text) < 120 and text not in seen:
            seen.append(text)
    return seen[:limit]


def call_reasoning_llm(messages: list[dict[str, str]], max_tokens: int = 900) -> str:
    payload = post_json(
        HF_CHAT_ENDPOINT,
        {
            "model": HF_REASONING_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        },
        {"authorization": f"Bearer {hf_token()}"},
    )
    return payload["choices"][0]["message"]["content"]


def get_profile_insights(profile: dict[str, Any]) -> dict[str, Any]:
    if not reasoning_enabled():
        return {"mode": "disabled", "model": None, "insights": None}

    text = profile_text(profile)
    cache = read_json(LLM_INSIGHTS_CACHE, {"entries": {}})
    key = f"profile:{HF_REASONING_MODEL}:{sha_key(text)}"
    cached = cache["entries"].get(key)
    if cached:
        age_days = (time.time() - datetime.fromisoformat(cached["generatedAt"]).timestamp()) / 86400
        if age_days < REASONING_REFRESH_DAYS:
            return {"mode": "cached", "model": HF_REASONING_MODEL, "insights": cached["insights"]}

    try:
        content = call_reasoning_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a PhD course recommendation analyst for Danish universities. "
                        "Expand a student's profile into useful course-search concepts. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Student profile:\n{text}\n\n"
                        "Return JSON with keys: inferredSubjects, adjacentSubjects, methods, "
                        "searchKeywords, genericSkills, rationale."
                    ),
                },
            ]
        )
        raw = parse_model_json(content)
        insights = {
            "inferredSubjects": clean_list(raw.get("inferredSubjects"), 8),
            "adjacentSubjects": clean_list(raw.get("adjacentSubjects"), 8),
            "methods": clean_list(raw.get("methods"), 8),
            "searchKeywords": clean_list(raw.get("searchKeywords"), 14),
            "genericSkills": clean_list(raw.get("genericSkills"), 6),
            "rationale": summarize(str(raw.get("rationale", "")), 360),
        }
        cache["entries"][key] = {
            "model": HF_REASONING_MODEL,
            "generatedAt": now_iso(),
            "insights": insights,
        }
        write_json(LLM_INSIGHTS_CACHE, cache)
        return {"mode": "fresh", "model": HF_REASONING_MODEL, "insights": insights}
    except Exception as error:
        return {"mode": "fallback", "model": HF_REASONING_MODEL, "warning": str(error), "insights": None}


def augment_profile(profile: dict[str, Any], insights: dict[str, Any] | None) -> dict[str, Any]:
    if not insights:
        return profile
    search_terms = (
        insights.get("searchKeywords", [])
        + insights.get("inferredSubjects", [])
        + insights.get("adjacentSubjects", [])
        + insights.get("methods", [])
    )
    augmented = dict(profile)
    augmented["llmSubjects"] = ", ".join(insights.get("inferredSubjects", []))
    augmented["llmAdjacentSubjects"] = ", ".join(insights.get("adjacentSubjects", []))
    augmented["llmSearchKeywords"] = ", ".join(insights.get("searchKeywords", []))
    augmented["llmMethods"] = ", ".join(insights.get("methods", []))
    augmented["keywords"] = ", ".join([profile.get("keywords", ""), *search_terms]).strip(", ")
    return augmented


def extract_terms(value: str | None) -> list[str]:
    if not value:
        return []
    phrases = [normalize(part) for part in re.split(r"[,;\n]", value) if " " in normalize(part)]
    tokens = [token for token in normalize(value).split() if len(token) > 2 and token not in STOP_WORDS]
    return list(dict.fromkeys([*phrases, *tokens]))


def includes_term(haystack: str, term: str) -> bool:
    term = normalize(term)
    if not term:
        return False
    if " " in term:
        return term in haystack
    return term in haystack.split()


def weighted_profile_signals(profile: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [
        ("research direction", profile.get("researchDirection"), 7),
        ("interests", profile.get("interests"), 7),
        ("project topic", profile.get("topic"), 5),
        ("study programme", profile.get("studyProgram"), 5),
        ("LLM inferred subjects", profile.get("llmSubjects"), 8),
        ("LLM adjacent subjects", profile.get("llmAdjacentSubjects"), 7),
        ("LLM search phrases", profile.get("llmSearchKeywords"), 7),
        ("LLM suggested methods", profile.get("llmMethods"), 6),
        ("methods and keywords", profile.get("keywords") or profile.get("methods"), 4),
    ]
    signals = []
    for label, value, weight in sources:
        for term in extract_terms(value):
            signals.append({"label": label, "term": term, "weight": weight})
    return sorted(signals, key=lambda item: len(item["term"]), reverse=True)


def recommend_lexical(profile: dict[str, Any], courses: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    signals = weighted_profile_signals(profile)
    selected_field_terms = FIELD_KEYWORDS.get(profile.get("area"), [])
    ranked = []

    for course in (course for course in courses if is_upcoming(course) and course_allowed_by_profile(profile, course)):
        haystack = normalize(
            " ".join(
                [
                    course.get("title", ""),
                    (course.get("courseSummary") or {}).get("shortSummary", ""),
                    (course.get("courseSummary") or {}).get("description", ""),
                    " ".join((course.get("courseSummary") or {}).get("learningOutcomes") or []),
                    (course.get("courseSummary") or {}).get("prerequisites", ""),
                    (course.get("courseSummary") or {}).get("teachingMethods", ""),
                    (course.get("courseSummary") or {}).get("audience", ""),
                    " ".join((course.get("courseSummary") or {}).get("keywords") or []),
                    course.get("description", ""),
                    course.get("fullDescription", ""),
                    course.get("phdSchool", ""),
                    course.get("university", ""),
                    course.get("city", ""),
                    course.get("startDate", ""),
                ]
            )
        )
        score = 0
        reasons = []
        reason_terms: dict[str, list[str]] = {}
        matched_phrases = []

        for signal in signals:
            term = signal["term"]
            if " " not in term and any(term in phrase for phrase in matched_phrases):
                continue
            if includes_term(haystack, term):
                score += signal["weight"] + (2 if " " in term else 0)
                if " " in term:
                    matched_phrases.append(term)
                reason_terms.setdefault(signal["label"], []).append(term)

        for label, terms in reason_terms.items():
            reasons.append(f"{label}: {', '.join(list(dict.fromkeys(terms))[:3])}")

        field_hits = [term for term in selected_field_terms if includes_term(haystack, term)]
        if field_hits:
            score += len(field_hits) * 6
            reasons.append(f"matches {', '.join(field_hits[:3])}")

        school_or_university_filter_active = bool(
            selected_universities(profile)
            or profile.get("includeOtherSchools")
            or profile.get("allowedOtherSchools")
        )
        if (
            not school_or_university_filter_active
            and profile.get("school")
            and profile["school"].lower() in course.get("phdSchool", "").lower()
        ):
            score += 14
            reasons.append("offered by your PhD school")
        elif (
            not school_or_university_filter_active
            and profile.get("university")
            and profile["university"].lower() in course.get("university", "").lower()
        ):
            score += 7
            reasons.append("same university")

        if profile.get("location") and profile["location"].lower() in course.get("city", "").lower():
            score += 5
            reasons.append("near your preferred location")

        if profile.get("minEcts") and course.get("ectsValue"):
            if course["ectsValue"] >= float(profile["minEcts"]):
                score += 2

        if "generic course" in course.get("description", "").lower() and not profile.get("includeGeneric"):
            score -= 7

        if score > 0:
            ranked.append({**course, "score": score, "reasons": reasons or ["text similarity with your profile"]})

    return sorted(ranked, key=lambda item: (-item["score"], item["title"]))[:limit]


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(vectors[0]))]


def normalize_embedding_output(output: Any, expected_count: int) -> list[list[float]]:
    if isinstance(output, list) and output and isinstance(output[0], (int, float)):
        return [[float(value) for value in output]]
    if isinstance(output, list) and output and isinstance(output[0], list):
        if output[0] and isinstance(output[0][0], (int, float)):
            if expected_count > 1 and len(output) == expected_count:
                return [[float(value) for value in vector] for vector in output]
            return [average_vectors(output)]
        if output[0] and isinstance(output[0][0], list):
            return [average_vectors(token_vectors) for token_vectors in output]
    raise RuntimeError("Unsupported embedding response shape")


def get_embeddings(items: list[dict[str, str]]) -> dict[str, list[float]]:
    cache = read_json(EMBEDDING_CACHE, {"model": HF_EMBEDDING_MODEL, "vectors": {}})
    if cache.get("model") != HF_EMBEDDING_MODEL:
        cache = {"model": HF_EMBEDDING_MODEL, "vectors": {}}

    results = {}
    missing = []
    for item in items:
        key = f"{item['id']}:{sha_key(item['text'])}"
        if key in cache["vectors"]:
            results[item["id"]] = cache["vectors"][key]
        else:
            missing.append({**item, "key": key})

    for start in range(0, len(missing), 8):
        batch = missing[start : start + 8]
        payload = post_json(
            HF_FEATURE_ENDPOINT,
            {"inputs": [item["text"] for item in batch], "options": {"wait_for_model": True}},
            {"authorization": f"Bearer {hf_token()}"},
        )
        vectors = normalize_embedding_output(payload, len(batch))
        for item, vector in zip(batch, vectors):
            cache["vectors"][item["key"]] = vector
            results[item["id"]] = vector

    if missing:
        write_json(EMBEDDING_CACHE, cache)
    return results


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def recommend_semantic(profile: dict[str, Any], courses: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    upcoming = [
        course for course in courses if is_upcoming(course) and course_allowed_by_profile(profile, course)
    ]
    lexical = recommend_lexical(profile, upcoming, len(upcoming))
    lexical_by_id = {course["id"]: course for course in lexical}
    items = [{"id": "profile", "text": profile_text(profile)}] + [
        {"id": course["id"], "text": course_text(course)} for course in upcoming
    ]
    embeddings = get_embeddings(items)
    profile_vector = embeddings["profile"]
    ranked = []

    for course in upcoming:
        similarity = cosine(profile_vector, embeddings[course["id"]])
        semantic_score = max(0.0, similarity) * 100
        lexical_score = lexical_by_id.get(course["id"], {}).get("score", 0)
        score = round(semantic_score + min(lexical_score, 30))
        ranked.append(
            {
                **course,
                "score": score,
                "semanticScore": round(semantic_score, 2),
                "lexicalScore": lexical_score,
                "reasons": [
                    f"semantic fit: {similarity * 100:.1f}% via {HF_EMBEDDING_MODEL}",
                    *lexical_by_id.get(course["id"], {}).get("reasons", []),
                ],
            }
        )
    return sorted(ranked, key=lambda item: (-item["score"], item["title"]))[:limit]


def create_recommendations(profile: dict[str, Any], courses: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
    reasoning = get_profile_insights(profile)
    augmented = augment_profile(profile, reasoning.get("insights"))

    if not embedding_enabled():
        return {
            "mode": "lexical",
            "model": None,
            "reasoning": reasoning,
            "augmentedProfile": augmented,
            "recommendations": recommend_lexical(augmented, courses, limit),
        }

    try:
        return {
            "mode": "semantic",
            "model": HF_EMBEDDING_MODEL,
            "reasoning": reasoning,
            "augmentedProfile": augmented,
            "recommendations": recommend_semantic(augmented, courses, limit),
        }
    except Exception as error:
        return {
            "mode": "lexical-fallback",
            "model": HF_EMBEDDING_MODEL,
            "warning": str(error),
            "reasoning": reasoning,
            "augmentedProfile": augmented,
            "recommendations": recommend_lexical(augmented, courses, limit),
        }


def cluster_terms(profiles: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for profile in profiles:
        for field in ["area", "studyProgram", "researchDirection", "interests", "topic", "keywords"]:
            for term in extract_terms(profile.get(field)):
                counts[term] = counts.get(term, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]]


def cluster_subscribers() -> dict[str, Any]:
    subscribers = [item for item in read_json(SUBSCRIBERS_FILE, []) if item.get("active")]
    groups: dict[str, list[dict[str, Any]]] = {}
    for subscriber in subscribers:
        profile = subscriber.get("profile", {})
        key = profile.get("area") or profile.get("studyProgram") or "Unspecified"
        groups.setdefault(key, []).append(subscriber)

    clusters = []
    for index, (theme, users) in enumerate(groups.items(), start=1):
        profiles = [user.get("profile", {}) for user in users]
        clusters.append(
            {
                "id": f"cluster-{index}",
                "size": len(users),
                "themes": [theme, *cluster_terms(profiles)],
                "users": [
                    {
                        "email": user["email"],
                        "area": user.get("profile", {}).get("area", ""),
                        "studyProgram": user.get("profile", {}).get("studyProgram", ""),
                        "researchDirection": user.get("profile", {}).get("researchDirection", ""),
                        "interests": user.get("profile", {}).get("interests", ""),
                    }
                    for user in users
                ],
            }
        )
    return {
        "mode": "profile-field",
        "model": None,
        "reasoning": {
            "enabled": reasoning_enabled(),
            "model": HF_REASONING_MODEL if reasoning_enabled() else None,
        },
        "clusters": clusters,
    }


class AgentHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/metadata":
                cache = get_course_cache()
                self.send_json(
                    {
                        "universities": UNIVERSITIES,
                        "phdSchools": get_phd_schools(cache.get("courses", [])),
                        "fields": list(FIELD_KEYWORDS.keys()),
                        "recommender": {
                            "mode": "semantic" if embedding_enabled() else "lexical",
                            "model": HF_EMBEDDING_MODEL if embedding_enabled() else None,
                            "semanticEnabled": embedding_enabled(),
                        },
                        "reasoning": {
                            "enabled": reasoning_enabled(),
                            "model": HF_REASONING_MODEL if reasoning_enabled() else None,
                            "refreshDays": REASONING_REFRESH_DAYS,
                        },
                        "cache": {
                            "scrapedAt": cache.get("scrapedAt"),
                            "count": cache.get("count", 0),
                            "source": cache.get("source"),
                        },
                    }
                )
                return
            if parsed.path == "/api/courses":
                refresh = query.get("refresh", ["false"])[0] == "true"
                self.send_json(scrape_courses() if refresh else get_course_cache())
                return
            if parsed.path == "/api/clusters":
                self.send_json(cluster_subscribers())
                return
            return super().do_GET()
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/recommend":
                profile = self.read_body_json()
                cache = get_course_cache()
                if not cache.get("count"):
                    cache = scrape_courses(max_pages=8)
                result = create_recommendations(profile, cache["courses"], int(profile.get("limit", 8)))
                self.send_json(
                    {
                        "profile": profile,
                        "recommender": {
                            "mode": result["mode"],
                            "model": result["model"],
                            "warning": result.get("warning"),
                        },
                        "reasoning": result["reasoning"],
                        "cache": {
                            "scrapedAt": cache.get("scrapedAt"),
                            "count": cache.get("count", 0),
                            "source": cache.get("source"),
                        },
                        "recommendations": result["recommendations"],
                    }
                )
                return
            if self.path == "/api/subscribe":
                payload = self.read_body_json()
                email = payload.get("email", "")
                if "@" not in email:
                    self.send_json({"error": "A valid email address is required."}, 400)
                    return
                subscribers = read_json(SUBSCRIBERS_FILE, [])
                subscriber = {
                    "email": email,
                    "profile": payload.get("profile", {}),
                    "subscribedAt": now_iso(),
                    "active": True,
                }
                subscribers = [item for item in subscribers if item.get("email") != email]
                subscribers.append(subscriber)
                write_json(SUBSCRIBERS_FILE, subscribers)
                self.send_json({"subscriber": subscriber, "count": len(subscribers)})
                return
            self.send_json({"error": "API route not found"}, 404)
        except Exception as error:
            self.send_json({"error": str(error)}, 500)


def main() -> None:
    ensure_dirs()
    if "--scrape-once" in os.sys.argv:
        cache = scrape_courses(max_pages=int(os.getenv("MAX_SCRAPE_PAGES", "120")))
        print(f"Scraped {cache['count']} courses from {cache['totalPages']} pages.")
        return

    server = ThreadingHTTPServer(("127.0.0.1", PORT), AgentHandler)
    print(f"Python course recommendation agent running at http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
