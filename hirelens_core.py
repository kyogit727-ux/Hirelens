import re
from pathlib import Path

import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SKILL_ALIASES = {
    "JavaScript": ["javascript", "es6", "es6+", "ecmascript"],
    "React": ["react", "react.js", "reactjs", "mern", "mern stack"],
    "Node.js": ["node.js", "nodejs", "node js", "mern", "mern stack"],
    "Express": ["express", "express.js", "expressjs", "mern", "mern stack"],
    "REST APIs": ["rest api", "rest apis", "restful api", "restful apis", "rest service", "rest services"],
    "JSON": ["json"],
    "Databases": ["sql", "mysql", "postgresql", "postgres", "nosql", "mongodb", "mongo", "relational database", "mern"],
    "Git / GitHub": ["git", "github", "gitlab", "version control"],
    "Relevant CS/IT Education": ["computer science", "computer engineering", "information technology", "software engineering", "cse", "bca", "mca"],
    "TypeScript": ["typescript"],
    "Cloud": ["aws", "amazon web services", "gcp", "google cloud", "azure", "cloud deployment"],
    "Testing": ["jest", "mocha", "pytest", "cypress", "unit testing", "integration testing", "testing framework"],
    "Docker": ["docker", "containerization", "containerisation", "containerized", "containers"],
    "Agile / Scrum": ["agile", "scrum", "sprint", "sprints", "daily standup", "jira"],
    "Deployed Projects": ["deployed", "deployment", "hosted", "production", "vercel", "render", "netlify", "heroku", "github actions", "ci/cd"],
}

HEADING_GROUPS = {
    "responsibilities": {"key responsibilities", "responsibilities", "what you will do", "what you'll do", "role responsibilities"},
    "must": {"must-have skills", "must have skills", "required skills", "requirements", "minimum qualifications", "required qualifications", "essential skills"},
    "good": {"good-to-have skills", "good to have skills", "preferred skills", "preferred qualifications", "nice-to-have skills", "nice to have skills", "desirable skills", "bonus skills"},
    "soft": {"soft skills", "behavioral skills", "behavioural skills", "competencies"},
}


@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def clean_extracted_text(text):
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[•●▪◦‣]", "•", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join((page.get_text("text", sort=True) or "") for page in doc)
    doc.close()
    return clean_extracted_text(text)


def normalize(text):
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def phrase_exists(phrase, text):
    phrase, text = normalize(phrase), normalize(text)
    if re.fullmatch(r"[a-z0-9]+", phrase):
        return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None
    return phrase in text


def detect_skills(text):
    return [skill for skill, aliases in SKILL_ALIASES.items() if any(phrase_exists(a, text) for a in aliases)]


def heading_key(line):
    cleaned = normalize(re.sub(r"[^A-Za-z0-9 +/&'-]", "", line))
    for key, variants in HEADING_GROUPS.items():
        if cleaned in variants:
            return key
    return None


def extract_jd_sections(jd_text):
    sections = {"responsibilities": [], "must": [], "good": [], "soft": []}
    current = None
    for raw in jd_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = heading_key(line)
        if key:
            current = key
            continue
        if current and len(line) <= 60 and line.isupper() and not line.startswith("•"):
            current = None
            continue
        if current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def extract_jd_skill_groups(jd_text):
    sections = extract_jd_sections(jd_text)
    must = detect_skills(sections["must"])
    good = detect_skills(sections["good"])
    if not must:
        all_skills = detect_skills(jd_text)
        must = [s for s in all_skills if s not in good]
    good = [s for s in good if s not in must]
    return must, good, sections


def split_into_chunks(text, max_chunks=90):
    prepared = text.replace("•", "\n• ")
    pieces = re.split(r"\n+|(?<=[.!?])\s+", prepared)
    chunks = []
    for piece in pieces:
        piece = re.sub(r"^[\s•\-*–—]+", "", piece).strip()
        piece = re.sub(r"\s+", " ", piece)
        if 22 <= len(piece) <= 550:
            chunks.append(piece)
    return chunks[:max_chunks]


def build_jd_requirement_chunks(jd_text, sections):
    relevant = "\n".join(sections.get(k, "") for k in ("responsibilities", "must", "good", "soft") if sections.get(k, "").strip())
    chunks = split_into_chunks(relevant, 45)
    return chunks or split_into_chunks(jd_text, 45)


def candidate_name_from_resume(text, filename):
    for line in text.splitlines()[:8]:
        line = line.strip()
        if 3 <= len(line) <= 60 and "@" not in line and not re.search(r"\d{6,}", line) and not line.isupper():
            if 1 <= len(line.split()) <= 5:
                return line
    name = Path(filename).stem
    name = re.sub(r"^Resume[_\- ]*\d*[_\- ]*", "", name, flags=re.I)
    return name.replace("_", " ").replace("-", " ").strip()


def parsing_status(text):
    chars = len(re.sub(r"\s+", "", text))
    if chars < 120:
        return "Low text / possible scanned PDF"
    if chars < 350:
        return "Limited text"
    return "OK"


def default_skill_weight(skill, must_skills, good_skills):
    if skill in must_skills:
        return 5
    if skill in good_skills:
        return 2
    return 3


def weighted_keyword_score(resume_skills, jd_skills, skill_weights):
    active = [(s, float(skill_weights.get(s, 0))) for s in jd_skills if float(skill_weights.get(s, 0)) > 0]
    total = sum(w for _, w in active)
    if total == 0:
        return 0.0
    matched = sum(w for s, w in active if s in resume_skills)
    return 100.0 * matched / total


def semantic_score_and_evidence(model, jd_chunks, jd_embeddings, resume_text):
    resume_chunks = split_into_chunks(resume_text, 100)
    if not jd_chunks or not resume_chunks:
        return 0.0, []
    resume_embeddings = model.encode(resume_chunks, show_progress_bar=False)
    matrix = cosine_similarity(jd_embeddings, resume_embeddings)
    best_idx = matrix.argmax(axis=1)
    best_scores = matrix.max(axis=1)
    score = float(np.clip(best_scores.mean() * 100, 0, 100))
    evidence = []
    for i in np.argsort(best_scores)[::-1][:4]:
        evidence.append({
            "requirement": jd_chunks[int(i)],
            "resume_evidence": resume_chunks[int(best_idx[int(i)])],
            "similarity": float(best_scores[int(i)] * 100),
        })
    return score, evidence


def analyze_resume(model, jd_chunks, jd_embeddings, resume_file, must_skills, good_skills):
    text = extract_pdf_text(resume_file)
    resume_skills = detect_skills(text)
    semantic, evidence = semantic_score_and_evidence(model, jd_chunks, jd_embeddings, text)
    matched_must = [s for s in must_skills if s in resume_skills]
    missing_must = [s for s in must_skills if s not in resume_skills]
    matched_good = [s for s in good_skills if s in resume_skills]
    must_coverage = 100.0 * len(matched_must) / len(must_skills) if must_skills else 0.0
    good_coverage = 100.0 * len(matched_good) / len(good_skills) if good_skills else 0.0
    return {
        "Candidate": candidate_name_from_resume(text, resume_file.name),
        "Semantic Score": semantic,
        "Resume Skills": resume_skills,
        "Matched Must-Have": matched_must,
        "Missing Must-Have": missing_must,
        "Matched Good-to-Have": matched_good,
        "Must-Have Score": must_coverage,
        "Good-to-Have Score": good_coverage,
        "Semantic Evidence": evidence,
        "Parsing": parsing_status(text),
    }


def rerank(base_results, semantic_weight_pct, skill_weights, jd_skills):
    semantic_weight = semantic_weight_pct / 100.0
    results = []
    for base in base_results:
        c = dict(base)
        keyword = weighted_keyword_score(set(c["Resume Skills"]), jd_skills, skill_weights)
        c["Keyword Score"] = keyword
        c["Final Score"] = semantic_weight * c["Semantic Score"] + (1 - semantic_weight) * keyword
        results.append(c)
    results.sort(key=lambda x: x["Final Score"], reverse=True)
    for i, c in enumerate(results, 1):
        c["Rank"] = i
    return results


def explanation_for(c):
    parts = []
    if c["Semantic Score"] >= 70:
        parts.append("Strong contextual alignment with the role.")
    elif c["Semantic Score"] >= 50:
        parts.append("Moderate contextual alignment with the role.")
    else:
        parts.append("Limited contextual alignment with the role.")
    if c["Matched Must-Have"]:
        parts.append("Core matches: " + ", ".join(c["Matched Must-Have"][:6]) + ".")
    if c["Missing Must-Have"]:
        parts.append("Missing or undetected core skills: " + ", ".join(c["Missing Must-Have"][:5]) + ".")
    if c["Matched Good-to-Have"]:
        parts.append("Extra strengths: " + ", ".join(c["Matched Good-to-Have"][:5]) + ".")
    return " ".join(parts)


def compare_candidates(a, b):
    if a["Final Score"] == b["Final Score"]:
        return f"{a['Candidate']} and {b['Candidate']} have the same overall score."
    winner, loser = (a, b) if a["Final Score"] > b["Final Score"] else (b, a)
    semantic_gap = winner["Semantic Score"] - loser["Semantic Score"]
    keyword_gap = winner["Keyword Score"] - loser["Keyword Score"]
    driver = "semantic alignment" if semantic_gap > keyword_gap else "recruiter-weighted keyword coverage"
    return f"{winner['Candidate']} ranks above {loser['Candidate']} by {winner['Final Score']-loser['Final Score']:.1f} points. The larger advantage comes from {driver}."


def detect_jd_issues(jd_text, must_skills):
    text = normalize(jd_text)
    rules = {
        "young candidate": "Age-related wording can unnecessarily narrow the candidate pool.",
        "young and energetic": "Age-related wording can unnecessarily narrow the candidate pool.",
        "native english speaker": "Prefer a job-related language proficiency requirement.",
        "male candidate": "Gender-specific wording can exclude qualified applicants.",
        "female candidate": "Gender-specific wording can exclude qualified applicants.",
        "digital native": "This phrase can indirectly imply an age preference.",
        "rockstar": "Subjective wording makes criteria less objective.",
        "ninja": "Subjective wording makes criteria less objective.",
        "top-tier college": "Prestige-based wording can be unnecessarily restrictive.",
        "tier 1 college": "Prestige-based wording can be unnecessarily restrictive.",
    }
    issues = [(p, r) for p, r in rules.items() if p in text]
    if len(must_skills) >= 12:
        issues.append(("Large must-have list", "A very long mandatory list can exclude otherwise strong candidates."))
    return issues


def recruiter_answer(question, results):
    q = normalize(question)
    if not results:
        return "Run the ranking first."
    if "top 3" in q or "top three" in q:
        return "Top 3: " + "; ".join(f"#{c['Rank']} {c['Candidate']} ({c['Final Score']:.1f}%)" for c in results[:3])
    if "best" in q or "top candidate" in q:
        c = results[0]
        return f"{c['Candidate']} is ranked #1 at {c['Final Score']:.1f}%. {explanation_for(c)}"

    mentioned = [c for c in results if normalize(c["Candidate"]) in q]
    if len(mentioned) >= 2 and ("why" in q or "compare" in q or "above" in q):
        return compare_candidates(mentioned[0], mentioned[1])
    if len(mentioned) == 1:
        c = mentioned[0]
        if "missing" in q or "lack" in q:
            return "Missing or undetected must-have skills: " + (", ".join(c["Missing Must-Have"]) or "none")
        if "match" in q or "skill" in q:
            return "Matched must-have skills: " + (", ".join(c["Matched Must-Have"]) or "none")
        return f"{c['Candidate']} is ranked #{c['Rank']} at {c['Final Score']:.1f}%. {explanation_for(c)}"
    return "Ask about the top candidates, a candidate's missing skills, or why one named candidate ranks above another."
