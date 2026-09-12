import hashlib
import re

import pandas as pd
import streamlit as st

from hirelens_core import (
    analyze_resume,
    build_jd_requirement_chunks,
    compare_candidates,
    default_skill_weight,
    detect_jd_issues,
    explanation_for,
    extract_jd_skill_groups,
    extract_pdf_text,
    load_model,
    recruiter_answer,
    rerank,
)

st.set_page_config(page_title="HireLens", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 4rem;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:12px;}
</style>
""", unsafe_allow_html=True)

st.title("🎯 HireLens")
st.subheader("Explainable Semantic + Keyword Resume Shortlisting")
st.write("Upload one job description and a batch of resumes. HireLens ranks every candidate using semantic similarity plus recruiter-controlled skill matching — without asking an LLM to invent a score.")

m1, m2, m3 = st.columns(3)
m1.metric("Semantic matching", "Embeddings")
m2.metric("Keyword matching", "Recruiter weighted")
m3.metric("Explainability", "Top 3 + evidence")
st.divider()

left, right = st.columns(2)
with left:
    jd_file = st.file_uploader("📄 Job Description (PDF)", type=["pdf"], key="jd_file")
with right:
    resume_files = st.file_uploader("👥 Candidate Resumes (PDFs)", type=["pdf"], accept_multiple_files=True, key="resume_files")

jd_text = ""
must_skills, good_skills, sections = [], [], {}
all_jd_skills = []

if jd_file is not None:
    try:
        jd_text = extract_pdf_text(jd_file)
        signature = hashlib.sha1(jd_text.encode("utf-8")).hexdigest()
        if st.session_state.get("jd_signature") not in (None, signature):
            for key in ("base_results", "chat_history", "parse_warnings"):
                st.session_state.pop(key, None)
        st.session_state["jd_signature"] = signature
        must_skills, good_skills, sections = extract_jd_skill_groups(jd_text)
        all_jd_skills = list(dict.fromkeys(must_skills + good_skills))
    except Exception as exc:
        st.error("Could not read the JD PDF.")
        st.exception(exc)

if jd_text:
    st.divider()
    st.header("🧩 Skills understood from the JD")
    st.caption("Recruiters can inspect and change the importance of every identified skill. 0 = ignore, 5 = critical.")

    if all_jd_skills:
        cols = st.columns(2)
        for i, skill in enumerate(all_jd_skills):
            category = "Must-have" if skill in must_skills else "Good-to-have"
            default = default_skill_weight(skill, must_skills, good_skills)
            safe_key = re.sub(r"[^a-z0-9]+", "_", skill.lower()).strip("_")
            with cols[i % 2]:
                st.slider(f"{skill} · {category}", 0, 5, default, 1, key=f"skill_weight_{safe_key}")
    else:
        st.warning("No known technical skills were detected in the JD. Semantic ranking can still be used.")

    st.subheader("🎚️ Semantic ↔ Keyword Ranking Balance")
    semantic_weight_pct = st.slider(
        "How much should the final ranking depend on semantic meaning?",
        0, 100, 45, 5,
        key="semantic_weight_pct",
        help="0% = keyword-only; 100% = semantic-only; values in between combine both."
    )
    a, b = st.columns(2)
    a.metric("Semantic", f"{semantic_weight_pct}%")
    b.metric("Keyword", f"{100-semantic_weight_pct}%")
else:
    semantic_weight_pct = st.session_state.get("semantic_weight_pct", 45)

analyze = st.button("🚀 Analyze & Rank Candidates", type="primary", use_container_width=True)

if analyze:
    if not jd_text:
        st.error("Upload a Job Description PDF first.")
    elif not resume_files:
        st.error("Upload at least one resume.")
    else:
        try:
            with st.spinner("Loading semantic model and preparing the JD..."):
                model = load_model()
                jd_chunks = build_jd_requirement_chunks(jd_text, sections)
                jd_embeddings = model.encode(jd_chunks, show_progress_bar=False)

            base_results, warnings = [], []
            progress = st.progress(0)
            status = st.empty()
            for i, resume in enumerate(resume_files):
                status.write(f"Analyzing {resume.name}...")
                item = analyze_resume(model, jd_chunks, jd_embeddings, resume, must_skills, good_skills)
                base_results.append(item)
                if item["Parsing"] != "OK":
                    warnings.append(f"{resume.name}: {item['Parsing']}")
                progress.progress((i + 1) / len(resume_files))
            progress.empty()
            status.empty()
            st.session_state["base_results"] = base_results
            st.session_state["must_skills"] = must_skills
            st.session_state["good_skills"] = good_skills
            st.session_state["all_jd_skills"] = all_jd_skills
            st.session_state["jd_text"] = jd_text
            st.session_state["parse_warnings"] = warnings
        except Exception as exc:
            st.error("Analysis failed.")
            st.exception(exc)

if "base_results" in st.session_state:
    must_skills = st.session_state.get("must_skills", [])
    good_skills = st.session_state.get("good_skills", [])
    all_jd_skills = st.session_state.get("all_jd_skills", [])

    current_weights = {}
    for skill in all_jd_skills:
        safe_key = re.sub(r"[^a-z0-9]+", "_", skill.lower()).strip("_")
        current_weights[skill] = st.session_state.get(
            f"skill_weight_{safe_key}", default_skill_weight(skill, must_skills, good_skills)
        )

    current_semantic = st.session_state.get("semantic_weight_pct", 45)
    results = rerank(st.session_state["base_results"], current_semantic, current_weights, all_jd_skills)

    st.success(f"Analysis complete — {len(results)} candidates ranked.")
    st.caption(f"Live mode: {current_semantic}% semantic + {100-current_semantic}% recruiter-weighted keyword matching. Adjust sliders above to rerank instantly.")

    with st.expander("🔍 Current JD skills and recruiter weights"):
        if all_jd_skills:
            st.dataframe(pd.DataFrame([
                {"Skill": s, "JD category": "Must-have" if s in must_skills else "Good-to-have", "Recruiter weight": current_weights[s]}
                for s in all_jd_skills
            ]), hide_index=True, use_container_width=True)

    if st.session_state.get("parse_warnings"):
        with st.expander("⚠️ Resume parsing warnings"):
            for warning in st.session_state["parse_warnings"]:
                st.write("• " + warning)

    st.header("🏆 Full Candidate Ranking")
    table = pd.DataFrame([{
        "Rank": c["Rank"],
        "Candidate": c["Candidate"],
        "Overall": round(c["Final Score"], 1),
        "Semantic": round(c["Semantic Score"], 1),
        "Weighted Keyword": round(c["Keyword Score"], 1),
        "Must-Have Coverage": round(c["Must-Have Score"], 1),
        "Good-to-Have Coverage": round(c["Good-to-Have Score"], 1),
        "Parsing": c["Parsing"],
    } for c in results])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.bar_chart(table.set_index("Candidate")[["Overall"]])
    st.download_button("⬇️ Download ranking as CSV", table.to_csv(index=False).encode(), "hirelens_rankings.csv", "text/csv")

    st.divider()
    st.header("🥇 Top 3 — Explainable Shortlist")
    for c in results[:3]:
        medal = {1:"🥇", 2:"🥈", 3:"🥉"}.get(c["Rank"], "")
        st.subheader(f"{medal} #{c['Rank']} — {c['Candidate']}")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Overall", f"{c['Final Score']:.1f}%")
        x2.metric("Semantic", f"{c['Semantic Score']:.1f}%")
        x3.metric("Weighted Keyword", f"{c['Keyword Score']:.1f}%")
        x4.metric("Must-Have Coverage", f"{c['Must-Have Score']:.1f}%")

        mc, missc = st.columns(2)
        with mc:
            st.markdown("**✅ Core skills matched**")
            for skill in c["Matched Must-Have"] or ["No core keyword matches detected."]:
                st.write(("✓ " if c["Matched Must-Have"] else "") + skill)
        with missc:
            st.markdown("**⚠️ Core skills missing / undetected**")
            for skill in c["Missing Must-Have"] or ["No core skill gaps detected."]:
                st.write(("• " if c["Missing Must-Have"] else "") + skill)

        if c["Matched Good-to-Have"]:
            st.markdown("**⭐ Good-to-have strengths:** " + ", ".join(c["Matched Good-to-Have"]))
        st.info(explanation_for(c))

        with st.expander("🔗 Show semantic evidence"):
            for ev in c["Semantic Evidence"]:
                st.markdown(f"**JD requirement ({ev['similarity']:.1f}% similarity)**")
                st.write(ev["requirement"])
                st.markdown("**Best resume evidence**")
                st.write(ev["resume_evidence"])
                st.divider()

    st.divider()
    st.header("⚖️ Candidate-vs-Candidate Explanation")
    names = [c["Candidate"] for c in results]
    ca, cb = st.columns(2)
    with ca:
        name_a = st.selectbox("Candidate A", names, index=0, key="compare_a")
    with cb:
        name_b = st.selectbox("Candidate B", names, index=min(1, len(names)-1), key="compare_b")

    cand_a = next(c for c in results if c["Candidate"] == name_a)
    cand_b = next(c for c in results if c["Candidate"] == name_b)
    st.dataframe(pd.DataFrame({
        "Metric": ["Overall", "Semantic", "Weighted Keyword", "Must-Have Coverage", "Good-to-Have Coverage"],
        name_a: [cand_a["Final Score"], cand_a["Semantic Score"], cand_a["Keyword Score"], cand_a["Must-Have Score"], cand_a["Good-to-Have Score"]],
        name_b: [cand_b["Final Score"], cand_b["Semantic Score"], cand_b["Keyword Score"], cand_b["Must-Have Score"], cand_b["Good-to-Have Score"]],
    }), hide_index=True, use_container_width=True)
    if name_a != name_b:
        st.info(compare_candidates(cand_a, cand_b))

    st.divider()
    st.header("🛡️ JD Bias / Over-Narrow Wording Check")
    issues = detect_jd_issues(st.session_state.get("jd_text", ""), must_skills)
    if issues:
        for phrase, reason in issues:
            st.warning(f"**{phrase}** — {reason}")
    else:
        st.success("No obvious age-, gender-, prestige-, or subjective restrictive wording was detected by the rule-based check.")
    st.caption("Screening aid only; not a legal determination of discrimination or compliance.")

    st.divider()
    st.header("💬 Recruiter Assistant")
    st.write("Ask about rankings, matched/missing skills, or why one candidate ranks above another. The assistant reads already-computed ranking data; it does not create the scores.")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for item in st.session_state["chat_history"]:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    question = st.chat_input("Example: Why is Aditi Sharma ranked above Rohan Verma?")
    if question:
        answer = recruiter_answer(question, results)
        st.session_state["chat_history"].extend([
            {"role":"user","content":question},
            {"role":"assistant","content":answer}
        ])
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    st.divider()
    with st.expander("🧠 How HireLens works — judging walkthrough"):
        st.markdown("""
1. **PDF parsing:** PyMuPDF extracts and normalizes text from the JD and resumes.
2. **JD understanding:** Must-have and good-to-have sections are detected, then known technologies are mapped to canonical skills using aliases.
3. **Keyword matching:** Each detected JD skill receives a recruiter-controlled weight from 0–5. Keyword score = matched weight points / total active weight points.
4. **Semantic matching:** JD requirements and resume chunks are encoded with `all-MiniLM-L6-v2`. Cosine similarity finds the best resume evidence for each JD requirement.
5. **Hybrid ranking:** `Final Score = S × Semantic + (1−S) × Weighted Keyword`, where S is controlled by the semantic/keyword slider.
6. **Explainability:** The top 3 show matched skills, missing skills, component scores, and actual semantic evidence.
7. **Bonus features:** Candidate comparison, recruiter Q&A, JD wording checks, CSV export, and defensive PDF parsing.

**No LLM is asked to assign the candidate score.**
        """)

st.divider()
st.caption("HireLens • Explainable semantic + keyword resume shortlisting")
