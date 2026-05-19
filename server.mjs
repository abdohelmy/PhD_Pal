import { createServer } from "node:http";
import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createHash } from "node:crypto";
import { extname, join, resolve } from "node:path";
import { URL } from "node:url";

const ROOT = resolve(".");
const PUBLIC_DIR = join(ROOT, "public");
const DATA_DIR = join(ROOT, "data");
const COURSE_CACHE = join(DATA_DIR, "courses.json");
const SUBSCRIBERS_FILE = join(DATA_DIR, "subscribers.json");
const EMBEDDING_CACHE = join(DATA_DIR, "embeddings.json");
const LLM_INSIGHTS_CACHE = join(DATA_DIR, "llm-insights.json");
const OUTBOX_DIR = join(DATA_DIR, "outbox");

const SOURCE_BASE = "https://phdcourses.dk";
const USER_AGENT =
  "Danish PhD Course Recommendation Agent/0.1 (+local research prototype; contact owner)";
const PORT = Number(process.env.PORT || 3000);
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
const HF_EMBEDDING_MODEL =
  process.env.HF_EMBEDDING_MODEL || "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
const HF_FEATURE_ENDPOINT = `https://api-inference.huggingface.co/pipeline/feature-extraction/${HF_EMBEDDING_MODEL}`;
const HF_REASONING_MODEL = process.env.HF_REASONING_MODEL || "deepseek-ai/DeepSeek-R1:fastest";
const HF_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions";
const REASONING_REFRESH_MS =
  Number(process.env.REASONING_REFRESH_DAYS || 21) * 24 * 60 * 60 * 1000;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg"
};

const UNIVERSITIES = [
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
  "Aarhus University"
];

const FIELD_KEYWORDS = {
  "AI, data science, statistics": [
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "data science",
    "python",
    "statistics",
    "statistical",
    "bioinformatics",
    "causal",
    "stochastic",
    "llm",
    "large language model"
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
    "biomedicine",
    "molecular",
    "genome",
    "patient"
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
    "mechanical",
    "electrical",
    "chemical"
  ],
  "Humanities, design, arts": [
    "humanities",
    "arts",
    "history",
    "culture",
    "design",
    "architecture",
    "language",
    "literature",
    "writing",
    "communication",
    "media"
  ],
  "Social science, business, law": [
    "social",
    "business",
    "management",
    "economics",
    "law",
    "political",
    "organization",
    "organisation",
    "qualitative",
    "policy",
    "market"
  ],
  "Generic research skills": [
    "ethics",
    "responsible conduct",
    "research integrity",
    "teaching",
    "supervision",
    "project management",
    "academic writing",
    "literature search",
    "information searching",
    "career"
  ]
};

const STOP_WORDS = new Set([
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
  "support"
]);

async function ensureDataDirs() {
  await mkdir(DATA_DIR, { recursive: true });
  await mkdir(OUTBOX_DIR, { recursive: true });
}

async function readJson(file, fallback) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return fallback;
    throw error;
  }
}

async function writeJson(file, data) {
  await ensureDataDirs();
  await writeFile(file, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

function decodeHtml(value = "") {
  return value
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();
}

function stripTags(value = "") {
  return decodeHtml(value.replace(/<[^>]*>/g, " "));
}

function normalizeText(value = "") {
  return value.toLowerCase().replace(/[^a-z0-9æøå]+/gi, " ").trim();
}

function summarize(text, maxLength = 280) {
  const clean = stripTags(text);
  if (clean.length <= maxLength) return clean;
  const cut = clean.slice(0, maxLength);
  return `${cut.slice(0, Math.max(cut.lastIndexOf(" "), 140)).trim()}...`;
}

function parseEcts(value = "") {
  const match = decodeHtml(value).replace(",", ".").match(/(\d+(\.\d+)?)/);
  return match ? Number(match[1]) : null;
}

function includesSearchTerm(haystack, term) {
  const normalizedTerm = normalizeText(term);
  if (!normalizedTerm) return false;
  if (normalizedTerm.includes(" ")) return haystack.includes(normalizedTerm);
  return haystack.split(" ").includes(normalizedTerm);
}

function isUpcomingCourse(course) {
  const value = course.startDate || "";
  const year = value.match(/\b(20\d{2})\b/)?.[1];
  if (!year) return true;

  const monthNames = [
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
    "december"
  ];
  const monthIndex = monthNames.findIndex((month) => value.toLowerCase().includes(month));
  const now = new Date();
  const courseYear = Number(year);

  if (courseYear > now.getFullYear()) return true;
  if (courseYear < now.getFullYear()) return false;
  if (monthIndex === -1) return true;
  return monthIndex >= now.getMonth();
}

function parseCourseRows(html) {
  const courses = [];
  const tbodyPattern = /<tbody>\s*<tr>\s*<td[^>]*>\s*<a class="subtitle" href="([^"]+)">([\s\S]*?)<\/a>\s*<\/td>\s*<td>\s*([\s\S]*?)<\/td>\s*<td>\s*([\s\S]*?)<\/td>\s*<td>\s*([\s\S]*?)<\/td>\s*<\/tr>\s*<tr>\s*<td colspan="4">\s*<a href="\/phdSchool\/(\d+)"[\s\S]*?>([\s\S]*?)<\/a>\s*<p[^>]*>([\s\S]*?)<\/p>\s*<\/td>\s*<\/tr>\s*<\/tbody>/gi;
  let match;

  while ((match = tbodyPattern.exec(html))) {
    const [, href, title, city, startDate, ects, phdSchoolId, phdSchool, description] =
      match;
    const id = href.match(/\/Course\/(\d+)/)?.[1] || href;
    courses.push({
      id,
      title: decodeHtml(title),
      sourceUrl: new URL(href, SOURCE_BASE).toString(),
      city: decodeHtml(city),
      startDate: decodeHtml(startDate),
      ects: decodeHtml(ects),
      ectsValue: parseEcts(ects),
      phdSchoolId,
      phdSchool: decodeHtml(phdSchool),
      university: inferUniversity(decodeHtml(phdSchool)),
      description: summarize(description, 520),
      scrapedAt: new Date().toISOString()
    });
  }

  return courses;
}

function textAfterHeading(html, heading) {
  const pattern = new RegExp(
    `<h3[^>]*>\\s*${heading}\\s*<\\/h3>\\s*([\\s\\S]*?)(?=<h3|<h2|<\\/main|<div id="show-cookie)`,
    "i"
  );
  return stripTags(html.match(pattern)?.[1] || "");
}

function parseCourseDetail(html, fallback) {
  const title = stripTags(html.match(/<h2[^>]*>\s*([\s\S]*?)\s*<\/h2>/i)?.[1] || "");
  const phdSchool = stripTags(html.match(/<h3[^>]*>\s*([\s\S]*?)\s*<\/h3>/i)?.[1] || "");
  const descriptionBlock =
    html.match(/<h3[^>]*>[\s\S]*?<\/h3>\s*([\s\S]*?)\s*<a[^>]*>\s*Back\s*<\/a>/i)?.[1] ||
    "";
  const registrationHref =
    html.match(/<h3[^>]*>\s*Link\s*<\/h3>\s*<a[^>]*href="([^"]+)"/i)?.[1] || "";
  const fullDescription = stripTags(descriptionBlock || fallback.description || "");

  return {
    ...fallback,
    title: title || fallback.title,
    phdSchool: phdSchool || fallback.phdSchool,
    university: inferUniversity(phdSchool || fallback.phdSchool),
    description: summarize(fullDescription, 620),
    fullDescription,
    startDate: textAfterHeading(html, "Course dates") || fallback.startDate,
    lecturer: textAfterHeading(html, "Lecturer"),
    venue: textAfterHeading(html, "Place/Venue"),
    city: textAfterHeading(html, "City") || fallback.city,
    ects: textAfterHeading(html, "ECTS") || fallback.ects,
    registrationUrl: registrationHref ? decodeHtml(registrationHref) : fallback.registrationUrl,
    rawCourseText: stripTags(
      [
        title,
        phdSchool,
        descriptionBlock,
        textAfterHeading(html, "Course dates"),
        textAfterHeading(html, "Lecturer"),
        textAfterHeading(html, "Place/Venue"),
        textAfterHeading(html, "City"),
        textAfterHeading(html, "ECTS")
      ].join(" ")
    )
  };
}

async function enrichCourseDetails(course) {
  try {
    const html = await fetchHtml(course.sourceUrl);
    return parseCourseDetail(html, course);
  } catch {
    return course;
  }
}

async function enrichRecommendations(recommendations, profile = {}) {
  const enriched = [];
  for (const course of recommendations) {
    const detail = await enrichCourseDetails(course);
    enriched.push({
      ...detail,
      score: course.score,
      reasons: course.reasons,
      summary: buildCourseSummary(detail, profile)
    });
  }
  return enriched;
}

function fallbackCourseSummary(course = {}) {
  const text = course.fullDescription || course.description || "";
  const keywords = [
    ...extractSignalTerms(course.title),
    ...extractSignalTerms(course.phdSchool),
    ...extractSignalTerms(text)
  ].slice(0, 12);
  return {
    mode: "extractive",
    model: null,
    generatedAt: new Date().toISOString(),
    shortSummary: summarize(text, 360),
    description: summarize(text, 700),
    learningOutcomes: [],
    prerequisites: "",
    teachingMethods: "",
    audience: "",
    keywords: [...new Set(keywords)]
  };
}

function normalizeCourseSummary(raw = {}, course = {}) {
  return {
    mode: "llm",
    model: HF_REASONING_MODEL,
    generatedAt: new Date().toISOString(),
    shortSummary: summarize(raw.shortSummary || raw.summary || course.description || "", 360),
    description: summarize(raw.description || course.fullDescription || course.description || "", 900),
    learningOutcomes: cleanArray(raw.learningOutcomes || raw.outcomes, 8),
    prerequisites: summarize(raw.prerequisites || "", 240),
    teachingMethods: summarize(raw.teachingMethods || raw.methods || "", 240),
    audience: summarize(raw.audience || "", 220),
    keywords: cleanArray(raw.keywords, 14)
  };
}

function courseDetailText(course = {}) {
  return [
    `Title: ${course.title || ""}`,
    `PhD school: ${course.phdSchool || ""}`,
    `University: ${course.university || ""}`,
    `ECTS: ${course.ects || ""}`,
    `Dates: ${course.startDate || ""}`,
    `City: ${course.city || ""}`,
    `Lecturer: ${course.lecturer || ""}`,
    `Venue: ${course.venue || ""}`,
    `Full course text: ${course.rawCourseText || course.fullDescription || course.description || ""}`
  ].join("\n");
}

async function summarizeCourseWithLlm(course = {}) {
  const text = courseDetailText(course);
  const cache = await readJson(LLM_INSIGHTS_CACHE, { entries: {} });
  const key = `course-summary:${HF_REASONING_MODEL}:${hashText(text)}`;
  const cached = cache.entries[key];
  if (cached) return cached.summary;

  if (!reasoningEnabled()) {
    const summary = fallbackCourseSummary(course);
    cache.entries[key] = { generatedAt: new Date().toISOString(), summary };
    await writeJson(LLM_INSIGHTS_CACHE, cache);
    return summary;
  }

  try {
    const content = await callReasoningModel(
      [
        {
          role: "system",
          content:
            "You summarize PhD course pages for a course recommendation system. Return only valid JSON."
        },
        {
          role: "user",
          content: `Course page text:\n${text.slice(0, 9000)}\n\nReturn JSON with keys: shortSummary, description, learningOutcomes, prerequisites, teachingMethods, audience, keywords. Focus on what a PhD student needs to decide whether the course is relevant.`
        }
      ],
      { maxTokens: 1100 }
    );
    const summary = normalizeCourseSummary(parseJsonFromModel(content), course);
    cache.entries[key] = { generatedAt: new Date().toISOString(), summary };
    await writeJson(LLM_INSIGHTS_CACHE, cache);
    return summary;
  } catch (error) {
    console.warn(`Course summary failed for ${course.id}:`, error.message);
    return fallbackCourseSummary(course);
  }
}

async function enrichAndSummarizeCourse(course) {
  const detail = await enrichCourseDetails(course);
  const courseSummary = await summarizeCourseWithLlm(detail);
  return {
    ...detail,
    courseSummary,
    description: courseSummary.shortSummary || detail.description
  };
}

function inferUniversity(phdSchool) {
  const lower = phdSchool.toLowerCase();
  if (lower.includes("aarhus")) return "Aarhus University";
  if (lower.includes("aalborg")) return "Aalborg University";
  if (lower.includes("southern denmark") || lower.includes("sdu")) {
    return "University of Southern Denmark";
  }
  if (lower.includes("copenhagen") || lower.includes("ucph")) {
    if (lower.includes("business") || lower.includes("cbs")) {
      return "Copenhagen Business School";
    }
    return "University of Copenhagen";
  }
  if (lower.includes("dtu")) return "Technical University of Denmark";
  if (lower.includes("roskilde")) return "Roskilde University";
  if (lower.includes("greenland")) return "University of Greenland";
  if (lower.includes("royal danish academy")) return "The Royal Danish Academy";
  if (lower.includes("design school kolding")) return "Design school Kolding";
  if (lower.includes("it university")) return "IT University of Copenhagen";
  return "";
}

function parseTotalPages(html) {
  const match = html.match(/<article class="pagination">[\s\S]*?of\s+(\d+)/i);
  return match ? Number(match[1]) : 1;
}

async function fetchHtml(url) {
  const response = await fetch(url, {
    headers: {
      "user-agent": USER_AGENT,
      accept: "text/html,application/xhtml+xml"
    }
  });
  if (!response.ok) {
    throw new Error(`Request failed ${response.status} for ${url}`);
  }
  return response.text();
}

async function scrapeCourses({ maxPages = Number(process.env.MAX_SCRAPE_PAGES || 120) } = {}) {
  const firstUrl = `${SOURCE_BASE}/?searchWord=`;
  const firstHtml = await fetchHtml(firstUrl);
  const totalPages = Math.min(parseTotalPages(firstHtml), maxPages);
  const courseMap = new Map();

  for (const course of parseCourseRows(firstHtml)) {
    courseMap.set(course.id, course);
  }

  for (let page = 2; page <= totalPages; page += 1) {
    const html = await fetchHtml(`${SOURCE_BASE}/?page=${page}&currentSearchWord=&`);
    for (const course of parseCourseRows(html)) {
      courseMap.set(course.id, course);
    }
  }

  const listedCourses = [...courseMap.values()].sort((a, b) =>
    `${a.startDate} ${a.title}`.localeCompare(`${b.startDate} ${b.title}`)
  );
  const courses = [];
  const shouldSummarizeDetails = process.env.SCRAPE_FULL_DETAILS !== "false";
  const detailLimit = Number(process.env.COURSE_DETAIL_LIMIT || listedCourses.length);

  for (const course of listedCourses) {
    if (shouldSummarizeDetails && courses.length < detailLimit) {
      courses.push(await enrichAndSummarizeCourse(course));
    } else {
      courses.push({
        ...course,
        courseSummary: fallbackCourseSummary(course)
      });
    }
  }

  const cache = {
    source: SOURCE_BASE,
    scrapedAt: new Date().toISOString(),
    totalPages,
    count: courses.length,
    courses
  };
  await writeJson(COURSE_CACHE, cache);
  return cache;
}

async function getCourseCache({ allowStale = true } = {}) {
  const cache = await readJson(COURSE_CACHE, null);
  if (cache || allowStale) {
    return cache || { source: SOURCE_BASE, scrapedAt: null, count: 0, courses: [] };
  }
  return scrapeCourses();
}

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean).map((value) => String(value).trim()))].sort((a, b) =>
    a.localeCompare(b)
  );
}

function getPhdSchoolsFromCourses(courses = []) {
  return uniqueSorted(courses.map((course) => course.phdSchool));
}

function selectedUniversities(profile = {}) {
  const selected = Array.isArray(profile.preferredUniversities)
    ? profile.preferredUniversities
    : [];
  if (profile.university) selected.push(profile.university);
  return uniqueSorted(selected);
}

function matchesOwnSchool(profile = {}, course = {}) {
  return Boolean(
    profile.school &&
      course.phdSchool &&
      course.phdSchool.toLowerCase().includes(String(profile.school).toLowerCase())
  );
}

function courseAllowedByProfile(profile = {}, course = {}) {
  const universities = selectedUniversities(profile);
  if (universities.length && !universities.includes(course.university)) {
    return false;
  }

  const allowedOtherSchools = Array.isArray(profile.allowedOtherSchools)
    ? profile.allowedOtherSchools
    : [];
  if (!profile.includeOtherSchools && profile.school) {
    return matchesOwnSchool(profile, course);
  }
  if (profile.includeOtherSchools && allowedOtherSchools.length) {
    return matchesOwnSchool(profile, course) || allowedOtherSchools.includes(course.phdSchool);
  }

  return true;
}

function hashText(value) {
  return createHash("sha256").update(value).digest("hex").slice(0, 24);
}

function hfToken() {
  return process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || "";
}

function embeddingEnabled() {
  return Boolean(hfToken()) && process.env.RECOMMENDER_MODE !== "lexical";
}

function reasoningEnabled() {
  return Boolean(hfToken()) && process.env.REASONING_MODE !== "off";
}

function profileToEmbeddingText(profile = {}) {
  return [
    `Research area: ${profile.area || ""}`,
    `Study programme: ${profile.studyProgram || ""}`,
    `PhD school: ${profile.school || ""}`,
    `Research direction: ${profile.researchDirection || ""}`,
    `Interests and learning goals: ${profile.interests || ""}`,
    `Project topic: ${profile.topic || ""}`,
    `Methods and keywords: ${profile.keywords || profile.methods || ""}`,
    `LLM inferred subjects: ${profile.llmSubjects || ""}`,
    `LLM adjacent subjects: ${profile.llmAdjacentSubjects || ""}`,
    `LLM search phrases: ${profile.llmSearchKeywords || ""}`,
    `LLM suggested methods: ${profile.llmMethods || ""}`
  ]
    .filter((line) => !line.endsWith(": "))
    .join("\n");
}

function courseToEmbeddingText(course = {}) {
  const summary = course.courseSummary || {};
  return [
    `Course: ${course.title || ""}`,
    `PhD school: ${course.phdSchool || ""}`,
    `University: ${course.university || ""}`,
    `Curated summary: ${summary.shortSummary || ""}`,
    `Description: ${summary.description || course.description || ""}`,
    `Learning outcomes: ${(summary.learningOutcomes || []).join("; ")}`,
    `Prerequisites: ${summary.prerequisites || ""}`,
    `Teaching methods: ${summary.teachingMethods || ""}`,
    `Audience: ${summary.audience || ""}`,
    `Keywords: ${(summary.keywords || []).join(", ")}`,
    `ECTS: ${course.ects || ""}`,
    `Dates: ${course.startDate || ""}`,
    `City: ${course.city || ""}`
  ]
    .filter((line) => !line.endsWith(": "))
    .join("\n");
}

function averageVectors(vectors) {
  if (!vectors.length) return [];
  const totals = Array.from({ length: vectors[0].length }, () => 0);
  for (const vector of vectors) {
    for (let index = 0; index < totals.length; index += 1) {
      totals[index] += Number(vector[index] || 0);
    }
  }
  return totals.map((value) => value / vectors.length);
}

function normalizeEmbeddingOutput(output, expectedCount) {
  if (!Array.isArray(output)) {
    throw new Error("Unexpected Hugging Face embedding response.");
  }
  if (typeof output[0] === "number") return [output.map(Number)];
  if (Array.isArray(output[0]) && typeof output[0][0] === "number") {
    if (expectedCount > 1 && output.length === expectedCount) {
      return output.map((vector) => vector.map(Number));
    }
    return [averageVectors(output)];
  }
  if (Array.isArray(output[0]) && Array.isArray(output[0][0])) {
    return output.map((tokenVectors) => averageVectors(tokenVectors));
  }
  throw new Error("Unsupported Hugging Face embedding response shape.");
}

async function fetchEmbeddingsFromHuggingFace(texts) {
  const response = await fetch(HF_FEATURE_ENDPOINT, {
    method: "POST",
    headers: {
      authorization: `Bearer ${hfToken()}`,
      "content-type": "application/json"
    },
    body: JSON.stringify({
      inputs: texts,
      options: { wait_for_model: true }
    })
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `Hugging Face request failed: ${response.status}`);
  }
  return normalizeEmbeddingOutput(payload, texts.length);
}

async function getEmbeddings(items) {
  const cache = await readJson(EMBEDDING_CACHE, {
    model: HF_EMBEDDING_MODEL,
    vectors: {}
  });
  if (cache.model !== HF_EMBEDDING_MODEL) {
    cache.model = HF_EMBEDDING_MODEL;
    cache.vectors = {};
  }

  const results = new Map();
  const missing = [];

  for (const item of items) {
    const key = `${item.id}:${hashText(item.text)}`;
    if (cache.vectors[key]) {
      results.set(item.id, cache.vectors[key]);
    } else {
      missing.push({ ...item, key });
    }
  }

  for (let index = 0; index < missing.length; index += 8) {
    const batch = missing.slice(index, index + 8);
    const vectors = await fetchEmbeddingsFromHuggingFace(batch.map((item) => item.text));
    vectors.forEach((vector, vectorIndex) => {
      const item = batch[vectorIndex];
      cache.vectors[item.key] = vector;
      results.set(item.id, vector);
    });
  }

  if (missing.length) await writeJson(EMBEDDING_CACHE, cache);
  return results;
}

function cosineSimilarity(a = [], b = []) {
  let dot = 0;
  let aMagnitude = 0;
  let bMagnitude = 0;
  const length = Math.min(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    dot += a[index] * b[index];
    aMagnitude += a[index] * a[index];
    bMagnitude += b[index] * b[index];
  }
  if (!aMagnitude || !bMagnitude) return 0;
  return dot / (Math.sqrt(aMagnitude) * Math.sqrt(bMagnitude));
}

function parseJsonFromModel(content = "") {
  const withoutReasoning = content
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/```json/gi, "```")
    .trim();
  const fenced = withoutReasoning.match(/```([\s\S]*?)```/)?.[1];
  const candidate = fenced || withoutReasoning.match(/\{[\s\S]*\}/)?.[0] || withoutReasoning;
  return JSON.parse(candidate);
}

function cleanArray(value, limit = 10) {
  if (!Array.isArray(value)) return [];
  return [
    ...new Set(
      value
        .map((item) => stripTags(String(item)).trim())
        .filter((item) => item.length > 2 && item.length < 120)
    )
  ].slice(0, limit);
}

function normalizeProfileInsights(raw = {}) {
  return {
    inferredSubjects: cleanArray(raw.inferredSubjects, 8),
    adjacentSubjects: cleanArray(raw.adjacentSubjects, 8),
    methods: cleanArray(raw.methods, 8),
    searchKeywords: cleanArray(raw.searchKeywords, 14),
    genericSkills: cleanArray(raw.genericSkills, 6),
    rationale: summarize(raw.rationale || raw.why || "", 360)
  };
}

async function callReasoningModel(messages, { maxTokens = 900 } = {}) {
  const response = await fetch(HF_CHAT_ENDPOINT, {
    method: "POST",
    headers: {
      authorization: `Bearer ${hfToken()}`,
      "content-type": "application/json"
    },
    body: JSON.stringify({
      model: HF_REASONING_MODEL,
      messages,
      temperature: 0.2,
      max_tokens: maxTokens,
      stream: false
    })
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error?.message || payload.error || `Hugging Face chat failed: ${response.status}`);
  }
  return payload.choices?.[0]?.message?.content || "";
}

async function getProfileInsights(profile = {}) {
  if (!reasoningEnabled()) {
    return { mode: "disabled", model: null, insights: null };
  }

  const profileText = profileToEmbeddingText(profile);
  const key = `profile:${HF_REASONING_MODEL}:${hashText(profileText)}`;
  const cache = await readJson(LLM_INSIGHTS_CACHE, { entries: {} });
  const cached = cache.entries[key];
  if (cached && Date.now() - new Date(cached.generatedAt).getTime() < REASONING_REFRESH_MS) {
    return { mode: "cached", model: HF_REASONING_MODEL, insights: cached.insights };
  }

  try {
    const content = await callReasoningModel([
      {
        role: "system",
        content:
          "You are a PhD course recommendation analyst for Danish universities. Expand a student's profile into useful course-search concepts. Return only valid JSON."
      },
      {
        role: "user",
        content: `Student profile:\n${profileText}\n\nReturn JSON with keys: inferredSubjects, adjacentSubjects, methods, searchKeywords, genericSkills, rationale. Use concise academic terms, methods, and course subjects that may be relevant even if the student did not mention them explicitly.`
      }
    ]);
    const insights = normalizeProfileInsights(parseJsonFromModel(content));
    cache.entries[key] = {
      model: HF_REASONING_MODEL,
      generatedAt: new Date().toISOString(),
      insights
    };
    await writeJson(LLM_INSIGHTS_CACHE, cache);
    return { mode: "fresh", model: HF_REASONING_MODEL, insights };
  } catch (error) {
    console.warn("Reasoning profile expansion failed:", error.message);
    return { mode: "fallback", model: HF_REASONING_MODEL, warning: error.message, insights: null };
  }
}

function augmentProfileWithInsights(profile = {}, insights = null) {
  if (!insights) return profile;
  const join = (items) => items.filter(Boolean).join(", ");
  return {
    ...profile,
    llmSubjects: join(insights.inferredSubjects || []),
    llmAdjacentSubjects: join(insights.adjacentSubjects || []),
    llmSearchKeywords: join(insights.searchKeywords || []),
    llmMethods: join(insights.methods || []),
    llmGenericSkills: join(insights.genericSkills || []),
    keywords: join([
      profile.keywords || "",
      ...(insights.searchKeywords || []),
      ...(insights.inferredSubjects || []),
      ...(insights.adjacentSubjects || []),
      ...(insights.methods || [])
    ])
  };
}

function extractSignalTerms(value = "") {
  const phrases = String(value)
    .split(/[,;\n]/)
    .map((phrase) => normalizeText(phrase))
    .filter((phrase) => phrase.length > 3 && phrase.includes(" "));
  const tokens = normalizeText(value)
    .split(" ")
    .filter((token) => token.length > 2 && !STOP_WORDS.has(token));
  return [...new Set([...phrases, ...tokens])];
}

function weightedProfileSignals(profile = {}) {
  const sources = [
    { label: "research direction", value: profile.researchDirection, weight: 7 },
    { label: "interests", value: profile.interests, weight: 7 },
    { label: "project topic", value: profile.topic, weight: 5 },
    { label: "study programme", value: profile.studyProgram, weight: 5 },
    { label: "LLM inferred subjects", value: profile.llmSubjects, weight: 8 },
    { label: "LLM adjacent subjects", value: profile.llmAdjacentSubjects, weight: 7 },
    { label: "LLM search phrases", value: profile.llmSearchKeywords, weight: 7 },
    { label: "LLM suggested methods", value: profile.llmMethods, weight: 6 },
    { label: "methods and keywords", value: profile.keywords || profile.methods, weight: 4 }
  ];
  const signals = [];

  for (const source of sources) {
    for (const term of extractSignalTerms(source.value)) {
      signals.push({ ...source, term });
    }
  }

  return signals;
}

function recommendCourses(profile, courses, limit = 8) {
  const profileSignals = weightedProfileSignals(profile).sort(
    (a, b) => b.term.length - a.term.length
  );
  const selectedFieldTerms = FIELD_KEYWORDS[profile.area] || [];
  const preferredMonths = String(profile.months || "")
    .split(",")
    .map((month) => month.trim().toLowerCase())
    .filter(Boolean);

  return courses
    .filter(isUpcomingCourse)
    .filter((course) => courseAllowedByProfile(profile, course))
    .map((course) => {
      const haystack = normalizeText(
        [
          course.title,
          course.courseSummary?.shortSummary,
          course.courseSummary?.description,
          ...(course.courseSummary?.learningOutcomes || []),
          course.courseSummary?.prerequisites,
          course.courseSummary?.teachingMethods,
          course.courseSummary?.audience,
          ...(course.courseSummary?.keywords || []),
          course.description,
          course.fullDescription,
          course.phdSchool,
          course.university,
          course.city,
          course.startDate
        ].join(" ")
      );
      let score = 0;
      const reasons = [];
      const reasonSources = new Map();
      const matchedPhraseTerms = [];

      for (const signal of profileSignals) {
        if (
          !signal.term.includes(" ") &&
          matchedPhraseTerms.some((phrase) => phrase.includes(signal.term))
        ) {
          continue;
        }
        if (includesSearchTerm(haystack, signal.term)) {
          score += signal.term.includes(" ") ? signal.weight + 2 : signal.weight;
          if (signal.term.includes(" ")) matchedPhraseTerms.push(signal.term);
          if (!reasonSources.has(signal.label)) reasonSources.set(signal.label, []);
          reasonSources.get(signal.label).push(signal.term);
        }
      }

      for (const [label, terms] of reasonSources) {
        reasons.push(`${label}: ${[...new Set(terms)].slice(0, 3).join(", ")}`);
      }

      const fieldHits = selectedFieldTerms.filter((term) =>
        includesSearchTerm(haystack, term)
      );
      if (fieldHits.length) {
        score += fieldHits.length * 6;
        reasons.push(`matches ${fieldHits.slice(0, 3).join(", ")}`);
      }

      const schoolOrUniversityFilterActive =
        selectedUniversities(profile).length ||
        profile.includeOtherSchools ||
        (Array.isArray(profile.allowedOtherSchools) && profile.allowedOtherSchools.length);
      if (
        !schoolOrUniversityFilterActive &&
        profile.school &&
        course.phdSchool.toLowerCase().includes(profile.school.toLowerCase())
      ) {
        score += 14;
        reasons.push("offered by your PhD school");
      } else if (
        !schoolOrUniversityFilterActive &&
        profile.university &&
        course.university.toLowerCase().includes(profile.university.toLowerCase())
      ) {
        score += 7;
        reasons.push("same university");
      }

      if (profile.location && course.city.toLowerCase().includes(profile.location.toLowerCase())) {
        score += 5;
        reasons.push("near your preferred location");
      }

      if (
        preferredMonths.length &&
        preferredMonths.some((month) => course.startDate.toLowerCase().includes(month))
      ) {
        score += 4;
        reasons.push("fits your timing");
      }

      if (profile.minEcts && course.ectsValue && course.ectsValue >= Number(profile.minEcts)) {
        score += 2;
      }

      if (course.description.toLowerCase().includes("generic course")) {
        score -= profile.includeGeneric ? 0 : 7;
      }

      return {
        ...course,
        score,
        reasons: reasons.length ? reasons : ["text similarity with your profile"],
        summary: buildCourseSummary(course, profile)
      };
    })
    .filter((course) => course.score > 0)
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
    .slice(0, limit);
}

async function recommendCoursesWithEmbeddings(profile, courses, limit = 8) {
  const upcomingCourses = courses
    .filter(isUpcomingCourse)
    .filter((course) => courseAllowedByProfile(profile, course));
  const lexicalMatches = recommendCourses(profile, upcomingCourses, upcomingCourses.length);
  const lexicalById = new Map(lexicalMatches.map((course) => [course.id, course]));
  const profileText = profileToEmbeddingText(profile);
  const courseItems = upcomingCourses.map((course) => ({
    id: course.id,
    text: courseToEmbeddingText(course)
  }));
  const embeddings = await getEmbeddings([
    { id: "profile", text: profileText },
    ...courseItems
  ]);
  const profileVector = embeddings.get("profile");

  return upcomingCourses
    .map((course) => {
      const lexical = lexicalById.get(course.id);
      const similarity = cosineSimilarity(profileVector, embeddings.get(course.id));
      const semanticScore = Math.max(0, similarity) * 100;
      const lexicalScore = lexical?.score || 0;
      const score = Math.round(semanticScore + Math.min(lexicalScore, 30));
      const reasons = [
        `semantic fit: ${(similarity * 100).toFixed(1)}% via ${HF_EMBEDDING_MODEL}`,
        ...(lexical?.reasons || [])
      ];

      return {
        ...course,
        score,
        semanticScore: Number(semanticScore.toFixed(2)),
        lexicalScore,
        recommender: "hugging-face-embeddings",
        reasons,
        summary: buildCourseSummary(course, profile)
      };
    })
    .filter((course) => course.score > 0)
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title))
    .slice(0, limit);
}

async function createRecommendations(profile, courses, limit = 8) {
  const reasoning = await getProfileInsights(profile);
  const augmentedProfile = augmentProfileWithInsights(profile, reasoning.insights);

  if (!embeddingEnabled()) {
    return {
      mode: "lexical",
      model: null,
      reasoning,
      augmentedProfile,
      recommendations: recommendCourses(augmentedProfile, courses, limit)
    };
  }

  try {
    return {
      mode: "semantic",
      model: HF_EMBEDDING_MODEL,
      reasoning,
      augmentedProfile,
      recommendations: await recommendCoursesWithEmbeddings(augmentedProfile, courses, limit)
    };
  } catch (error) {
    console.warn("Semantic recommendations failed, using lexical fallback:", error.message);
    return {
      mode: "lexical-fallback",
      model: HF_EMBEDDING_MODEL,
      warning: error.message,
      reasoning,
      augmentedProfile,
      recommendations: recommendCourses(augmentedProfile, courses, limit)
    };
  }
}

function buildCourseSummary(course, profile = {}) {
  const area = profile.area ? ` for ${profile.area.toLowerCase()}` : "";
  const where = [course.city, course.university].filter(Boolean).join(", ");
  const date = course.startDate && course.startDate !== "-" ? ` Starts ${course.startDate}.` : "";
  const ects = course.ects ? ` Worth ${course.ects}.` : "";
  return `${course.title} looks relevant${area} because it covers ${course.description}${date}${ects}${
    where ? ` Location: ${where}.` : ""
  }`;
}

function htmlEscape(value = "") {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderDigest(profile, recommendations) {
  const profileContext = [
    profile.area,
    profile.studyProgram,
    profile.researchDirection,
    profile.interests
  ]
    .filter(Boolean)
    .join(" · ");
  const items = recommendations
    .map(
      (course) => `
        <li>
          <h3><a href="${htmlEscape(course.sourceUrl)}">${htmlEscape(course.title)}</a></h3>
          <p><strong>${htmlEscape(course.startDate || "Date TBA")}</strong> · ${htmlEscape(
            course.ects || "ECTS TBA"
          )} · ${htmlEscape(course.city || "Location TBA")}</p>
          <p>${htmlEscape(course.description)}</p>
          <p><em>Why:</em> ${htmlEscape(course.reasons.join(", "))}</p>
        </li>`
    )
    .join("");

  return `<!doctype html>
<html>
<body style="font-family: Arial, sans-serif; color: #17201b; line-height: 1.5;">
  <h1>Your Danish PhD course matches</h1>
  <p>Based on ${htmlEscape(profileContext || "your PhD profile")}, these upcoming courses look useful.</p>
  <ol>${items || "<li>No strong matches this week.</li>"}</ol>
  <p style="color:#5d6b63;font-size:13px;">Source: phdcourses.dk. Please confirm availability and registration rules on each course page.</p>
</body>
</html>`;
}

async function deliverDigest(subscriber, recommendations) {
  const subject = "Your weekly Danish PhD course recommendations";
  const html = renderDigest(subscriber.profile, recommendations);

  if (process.env.RESEND_API_KEY && process.env.FROM_EMAIL) {
    const response = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        authorization: `Bearer ${process.env.RESEND_API_KEY}`,
        "content-type": "application/json"
      },
      body: JSON.stringify({
        from: process.env.FROM_EMAIL,
        to: subscriber.email,
        subject,
        html
      })
    });
    if (!response.ok) {
      throw new Error(`Resend email failed: ${response.status} ${await response.text()}`);
    }
    return { mode: "resend", email: subscriber.email };
  }

  await ensureDataDirs();
  const safeEmail = subscriber.email.replace(/[^a-z0-9._-]+/gi, "_");
  const filename = `${new Date().toISOString().replace(/[:.]/g, "-")}_${safeEmail}.html`;
  await writeFile(join(OUTBOX_DIR, filename), html, "utf8");
  return { mode: "outbox", email: subscriber.email, file: join(OUTBOX_DIR, filename) };
}

async function runWeeklyDigest() {
  const cache = await getCourseCache({ allowStale: false });
  const subscribers = await readJson(SUBSCRIBERS_FILE, []);
  const deliveries = [];

  for (const subscriber of subscribers) {
    const recommendationResult = await createRecommendations(subscriber.profile, cache.courses, 6);
    const recommendations = await enrichRecommendations(
      recommendationResult.recommendations,
      recommendationResult.augmentedProfile || subscriber.profile
    );
    deliveries.push(await deliverDigest(subscriber, recommendations));
  }

  return {
    ranAt: new Date().toISOString(),
    subscriberCount: subscribers.length,
    courseCount: cache.count,
    deliveries
  };
}

function centroid(vectors) {
  return averageVectors(vectors);
}

function assignToCentroids(vector, centroids) {
  let bestIndex = 0;
  let bestScore = -Infinity;
  centroids.forEach((candidate, index) => {
    const score = cosineSimilarity(vector, candidate);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function clusterTerms(profiles) {
  const counts = new Map();
  for (const profile of profiles) {
    for (const term of [
      ...extractSignalTerms(profile.area),
      ...extractSignalTerms(profile.studyProgram),
      ...extractSignalTerms(profile.researchDirection),
      ...extractSignalTerms(profile.interests),
      ...extractSignalTerms(profile.topic),
      ...extractSignalTerms(profile.keywords)
    ]) {
      counts.set(term, (counts.get(term) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 8)
    .map(([term]) => term);
}

function userSummary(user) {
  const profile = user.profile || {};
  return {
    area: profile.area || "",
    studyProgram: profile.studyProgram || "",
    researchDirection: profile.researchDirection || "",
    interests: profile.interests || "",
    topic: profile.topic || "",
    keywords: profile.keywords || ""
  };
}

function kMeansUsers(users, vectors, k) {
  let centroids = users.slice(0, k).map((user) => vectors.get(user.email));
  let assignments = new Map();

  for (let iteration = 0; iteration < 12; iteration += 1) {
    assignments = new Map();
    users.forEach((user) => {
      const clusterIndex = assignToCentroids(vectors.get(user.email), centroids);
      if (!assignments.has(clusterIndex)) assignments.set(clusterIndex, []);
      assignments.get(clusterIndex).push(user);
    });

    centroids = centroids.map((oldCentroid, index) => {
      const clusterUsers = assignments.get(index) || [];
      if (!clusterUsers.length) return oldCentroid;
      return centroid(clusterUsers.map((user) => vectors.get(user.email)));
    });
  }

  return [...assignments.entries()]
    .map(([index, clusterUsers]) => ({
      id: `cluster-${index + 1}`,
      size: clusterUsers.length,
      themes: clusterTerms(clusterUsers.map((user) => user.profile || {})),
      users: clusterUsers.map((user) => ({
        email: user.email,
        ...userSummary(user)
      }))
    }))
    .sort((a, b) => b.size - a.size || a.id.localeCompare(b.id));
}

function normalizeClusterInsight(raw = {}) {
  return {
    label: summarize(raw.label || "", 80),
    expandedThemes: cleanArray(raw.expandedThemes, 8),
    searchKeywords: cleanArray(raw.searchKeywords, 12),
    courseCategories: cleanArray(raw.courseCategories, 8),
    digestStrategy: summarize(raw.digestStrategy || "", 220)
  };
}

async function reasonAboutCluster(cluster) {
  if (!reasoningEnabled() || !cluster.users.length) return cluster;

  const clusterText = JSON.stringify(
    {
      themes: cluster.themes,
      users: cluster.users.map(({ email, ...user }) => user)
    },
    null,
    2
  );
  const key = `cluster:${HF_REASONING_MODEL}:${hashText(clusterText)}`;
  const cache = await readJson(LLM_INSIGHTS_CACHE, { entries: {} });
  const cached = cache.entries[key];
  if (cached && Date.now() - new Date(cached.generatedAt).getTime() < REASONING_REFRESH_MS) {
    return { ...cluster, insight: cached.insight };
  }

  try {
    const content = await callReasoningModel([
      {
        role: "system",
        content:
          "You are optimizing course recommendations for clusters of PhD students. Return only valid JSON."
      },
      {
        role: "user",
        content: `Cluster data:\n${clusterText}\n\nReturn JSON with keys: label, expandedThemes, searchKeywords, courseCategories, digestStrategy. Infer adjacent course areas that may help this cluster beyond their exact wording.`
      }
    ]);
    const insight = normalizeClusterInsight(parseJsonFromModel(content));
    cache.entries[key] = {
      model: HF_REASONING_MODEL,
      generatedAt: new Date().toISOString(),
      insight
    };
    await writeJson(LLM_INSIGHTS_CACHE, cache);
    return { ...cluster, insight };
  } catch (error) {
    console.warn("Reasoning cluster labeling failed:", error.message);
    return { ...cluster, insightWarning: error.message };
  }
}

async function enrichClustersWithReasoning(clusters) {
  const enriched = [];
  for (const cluster of clusters) {
    enriched.push(await reasonAboutCluster(cluster));
  }
  return enriched;
}

async function clusterSubscribers() {
  const subscribers = (await readJson(SUBSCRIBERS_FILE, [])).filter((subscriber) => subscriber.active);
  if (!subscribers.length) {
    return {
      mode: embeddingEnabled() ? "semantic" : "profile-field",
      model: embeddingEnabled() ? HF_EMBEDDING_MODEL : null,
      reasoning: {
        enabled: reasoningEnabled(),
        model: reasoningEnabled() ? HF_REASONING_MODEL : null
      },
      clusters: []
    };
  }

  let result;
  if (!embeddingEnabled() || subscribers.length < 3) {
    const groups = new Map();
    for (const subscriber of subscribers) {
      const key = subscriber.profile?.area || subscriber.profile?.studyProgram || "Unspecified";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(subscriber);
    }
    result = {
      mode: "profile-field",
      model: null,
      clusters: [...groups.entries()].map(([theme, users], index) => ({
        id: `cluster-${index + 1}`,
        size: users.length,
        themes: [theme, ...clusterTerms(users.map((user) => user.profile || {}))].filter(Boolean),
        users: users.map((user) => ({
          email: user.email,
          ...userSummary(user)
        }))
      }))
    };
  } else {
    const embeddingItems = [];
    for (const subscriber of subscribers) {
      const reasoning = await getProfileInsights(subscriber.profile || {});
      const augmentedProfile = augmentProfileWithInsights(
        subscriber.profile || {},
        reasoning.insights
      );
      embeddingItems.push({
        id: subscriber.email,
        text: profileToEmbeddingText(augmentedProfile)
      });
    }
    const embeddings = await getEmbeddings(embeddingItems);
    const k = Math.min(6, Math.max(2, Math.round(Math.sqrt(subscribers.length))));
    result = {
      mode: "semantic",
      model: HF_EMBEDDING_MODEL,
      k,
      clusters: kMeansUsers(subscribers, embeddings, k)
    };
  }

  return {
    ...result,
    reasoning: {
      enabled: reasoningEnabled(),
      model: reasoningEnabled() ? HF_REASONING_MODEL : null,
      refreshDays: Math.round(REASONING_REFRESH_MS / (24 * 60 * 60 * 1000))
    },
    clusters: await enrichClustersWithReasoning(result.clusters)
  };
}

function readRequestBody(request) {
  return new Promise((resolveBody, reject) => {
    let body = "";
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        request.destroy();
        reject(new Error("Request body too large"));
      }
    });
    request.on("end", () => resolveBody(body));
    request.on("error", reject);
  });
}

function sendJson(response, statusCode, data) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(data, null, 2));
}

function sendError(response, error, statusCode = 500) {
  sendJson(response, statusCode, { error: error.message || String(error) });
}

async function handleApi(request, response, url) {
  if (url.pathname === "/api/metadata") {
    const cache = await getCourseCache();
    sendJson(response, 200, {
      universities: UNIVERSITIES,
      phdSchools: getPhdSchoolsFromCourses(cache.courses),
      fields: Object.keys(FIELD_KEYWORDS),
      recommender: {
        mode: embeddingEnabled() ? "semantic" : "lexical",
        model: embeddingEnabled() ? HF_EMBEDDING_MODEL : null,
        semanticEnabled: embeddingEnabled()
      },
      reasoning: {
        enabled: reasoningEnabled(),
        model: reasoningEnabled() ? HF_REASONING_MODEL : null,
        refreshDays: Math.round(REASONING_REFRESH_MS / (24 * 60 * 60 * 1000))
      },
      cache: {
        scrapedAt: cache.scrapedAt,
        count: cache.count,
        source: cache.source
      }
    });
    return;
  }

  if (url.pathname === "/api/courses") {
    const refresh = url.searchParams.get("refresh") === "true";
    const cache = refresh ? await scrapeCourses() : await getCourseCache();
    sendJson(response, 200, cache);
    return;
  }

  if (url.pathname === "/api/recommend" && request.method === "POST") {
    const profile = JSON.parse(await readRequestBody(request) || "{}");
    let cache = await getCourseCache();
    if (!cache.count) cache = await scrapeCourses({ maxPages: 8 });
    const recommendationResult = await createRecommendations(
      profile,
      cache.courses,
      Number(profile.limit || 8)
    );
    const recommendations = await enrichRecommendations(
      recommendationResult.recommendations,
      recommendationResult.augmentedProfile || profile
    );
    sendJson(response, 200, {
      profile,
      recommender: {
        mode: recommendationResult.mode,
        model: recommendationResult.model,
        warning: recommendationResult.warning
      },
      reasoning: {
        mode: recommendationResult.reasoning?.mode,
        model: recommendationResult.reasoning?.model,
        warning: recommendationResult.reasoning?.warning,
        insights: recommendationResult.reasoning?.insights
      },
      cache: {
        scrapedAt: cache.scrapedAt,
        count: cache.count,
        source: cache.source
      },
      recommendations
    });
    return;
  }

  if (url.pathname === "/api/subscribe" && request.method === "POST") {
    const payload = JSON.parse(await readRequestBody(request) || "{}");
    if (!payload.email || !payload.email.includes("@")) {
      sendJson(response, 400, { error: "A valid email address is required." });
      return;
    }
    const subscribers = await readJson(SUBSCRIBERS_FILE, []);
    const existing = subscribers.find((subscriber) => subscriber.email === payload.email);
    const subscriber = {
      email: payload.email,
      profile: payload.profile || {},
      subscribedAt: new Date().toISOString(),
      active: true
    };
    if (existing) Object.assign(existing, subscriber);
    else subscribers.push(subscriber);
    await writeJson(SUBSCRIBERS_FILE, subscribers);
    sendJson(response, 200, { subscriber, count: subscribers.length });
    return;
  }

  if (url.pathname === "/api/clusters") {
    sendJson(response, 200, await clusterSubscribers());
    return;
  }

  if (url.pathname === "/api/digest/run" && request.method === "POST") {
    sendJson(response, 200, await runWeeklyDigest());
    return;
  }

  if (url.pathname === "/api/outbox") {
    await ensureDataDirs();
    const files = await readdir(OUTBOX_DIR);
    sendJson(response, 200, { files: files.sort().reverse() });
    return;
  }

  sendJson(response, 404, { error: "API route not found." });
}

async function serveStatic(response, pathname) {
  const requestedPath = pathname === "/" ? "/index.html" : pathname;
  const filePath = resolve(PUBLIC_DIR, `.${requestedPath}`);
  if (!filePath.startsWith(PUBLIC_DIR)) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }

  const contentType = MIME_TYPES[extname(filePath)] || "application/octet-stream";
  response.writeHead(200, { "content-type": contentType });
  createReadStream(filePath)
    .on("error", () => {
      response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      response.end("Not found");
    })
    .pipe(response);
}

async function main() {
  await ensureDataDirs();

  if (process.argv.includes("--scrape-once")) {
    const cache = await scrapeCourses();
    console.log(`Scraped ${cache.count} courses from ${cache.totalPages} pages.`);
    return;
  }

  if (process.argv.includes("--digest-once")) {
    console.log(JSON.stringify(await runWeeklyDigest(), null, 2));
    return;
  }

  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, `http://${request.headers.host}`);
      if (url.pathname.startsWith("/api/")) {
        await handleApi(request, response, url);
      } else {
        await serveStatic(response, url.pathname);
      }
    } catch (error) {
      console.error(error);
      sendError(response, error);
    }
  });

  server.listen(PORT, () => {
    console.log(`Course recommendation agent running at http://localhost:${PORT}`);
  });

  setInterval(() => {
    runWeeklyDigest().catch((error) => console.error("Weekly digest failed", error));
  }, WEEK_MS).unref();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
