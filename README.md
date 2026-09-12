# HireLens

HireLens is an explainable AI-assisted resume ranking engine that ranks a batch of resumes against a Job Description using **semantic similarity** plus **recruiter-weighted keyword matching**.

The system does **not** ask an LLM to invent candidate scores. Ranking comes from deterministic skill matching, Sentence Transformer embeddings, cosine similarity, and a transparent weighted formula.

## Features

- Upload one Job Description PDF and multiple resume PDFs
- Extract and normalize PDF text with PyMuPDF
- Detect must-have and good-to-have JD sections
- Show the skills understood from the JD
- Recruiter-controlled skill importance sliders from **0 to 5**
- Recruiter-controlled **Semantic ↔ Keyword** balance slider
- Semantic matching with `all-MiniLM-L6-v2`
- Cosine-similarity evidence for JD requirements
- Full ranked candidate list with component scores
- Explainable top-3 shortlist with matched and missing skills
- Candidate-vs-candidate comparison
- Recruiter Q&A over already-computed ranking data
- Rule-based JD bias / overly narrow wording checks
- CSV ranking export
- Defensive handling of PDFs with limited extractable text

## Ranking Logic

### Semantic score
The JD and resumes are split into meaningful chunks and encoded with `all-MiniLM-L6-v2`. For every JD requirement, HireLens finds the most similar resume chunk using cosine similarity and aggregates the best matches.

### Recruiter-weighted keyword score
Explicit requirements are mapped to canonical skills using aliases such as `React.js → React`, `NodeJS → Node.js`, and `AWS/GCP/Azure → Cloud`.

Each detected JD skill gets an editable recruiter weight from `0` to `5`.

```text
Weighted Keyword Score = matched skill-weight points / total active skill-weight points
```

### Final score

```text
Final Score = S × Semantic Score + (1 − S) × Weighted Keyword Score
```

`S` is controlled by the recruiter using the Semantic ↔ Keyword slider.

- `0% semantic` = keyword-only ranking
- `100% semantic` = semantic-only ranking
- values in between = hybrid ranking

Changing recruiter controls re-ranks candidates without recomputing the expensive semantic embeddings.

## Project Structure

```text
Hirelens/
├── app.py
├── hirelens_core.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

The first semantic analysis may download the `all-MiniLM-L6-v2` model.

## Tech Stack

Python · Streamlit · PyMuPDF · Sentence Transformers · scikit-learn · pandas · NumPy

## Responsible Use

HireLens is a decision-support prototype, not a replacement for human hiring judgment. The JD wording check is heuristic and is not a legal determination of discrimination or compliance.
