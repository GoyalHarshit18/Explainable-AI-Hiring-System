"""
feature_engineering.py

Feature Engineering Module

Builds recruiter-aware features from parsed candidate data.
"""

import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# JD Keyword Extraction
# ==========================================================

STOPWORDS = {
    "the","and","or","with","for","to","of","in","on","at",
    "is","are","be","a","an","using","experience","years"
}


def extract_keywords(jd_text, min_len=3):

    words = re.findall(r"[a-zA-Z0-9+#.\-]+", jd_text.lower())

    keywords = []

    for word in words:

        if len(word) < min_len:
            continue

        if word in STOPWORDS:
            continue

        keywords.append(word)

    return sorted(set(keywords))


# ==========================================================
# Skill Extraction
# ==========================================================

def candidate_skill_set(parsed_skills):

    skills = set()

    for skill in parsed_skills:

        skills.add(
            skill["skill"].lower().strip()
        )

    return skills


# ==========================================================
# JD Skill Match
# ==========================================================

def jd_skill_overlap(candidate_skills,
                     jd_keywords):

    if len(candidate_skills) == 0:
        return 0

    overlap = 0

    for skill in candidate_skills:

        if skill in jd_keywords:
            overlap += 1

    return overlap


def matched_skill_ratio(candidate_skills,
                        jd_keywords):

    if len(jd_keywords) == 0:
        return 0

    overlap = jd_skill_overlap(
        candidate_skills,
        jd_keywords
    )

    return overlap / len(jd_keywords)


# ==========================================================
# Resume Keyword Match
# ==========================================================

def keyword_hits(resume_text,
                 jd_keywords):

    resume = resume_text.lower()

    count = 0

    for kw in jd_keywords:

        if kw in resume:
            count += 1

    return count


def keyword_density(resume_text,
                    jd_keywords):

    words = resume_text.split()

    if len(words) == 0:
        return 0

    hits = keyword_hits(
        resume_text,
        jd_keywords
    )

    return hits / len(words)


# ==========================================================
# Important Skill Coverage
# ==========================================================

IMPORTANT_SKILLS = {

    "python",

    "machine learning",

    "deep learning",

    "nlp",

    "retrieval",

    "ranking",

    "faiss",

    "bm25",

    "langchain",

    "sentence transformers",

    "transformers",

    "pytorch",

    "tensorflow",

    "docker",

    "kubernetes"

}


def important_skill_score(skill_set):

    if len(skill_set) == 0:
        return 0

    score = 0

    for skill in skill_set:

        if skill in IMPORTANT_SKILLS:

            score += 1

    return score / len(IMPORTANT_SKILLS)


# ==========================================================
# Resume Completeness
# ==========================================================

def resume_completeness(candidate):

    score = 0

    if candidate["career"]:
        score += 1

    if candidate["skills"]:
        score += 1

    if candidate["education"]:
        score += 1

    if candidate["certifications"]:
        score += 1

    return score / 4.0

# ==========================================================
# Role Suitability
# ==========================================================

GOOD_ROLES = {

    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "nlp engineer",
    "search engineer",
    "recommendation systems engineer",
    "data scientist",
    "backend engineer",
    "software engineer",
    "applied ml engineer",
    "ai research engineer",
    "senior data scientist",
    "research engineer"

}

BAD_ROLES = {

    "content writer",
    "customer support",
    "operations manager",
    "sales executive",
    "marketing manager",
    "graphic designer",
    "mechanical engineer",
    "civil engineer",
    "electrical engineer"

}


def role_suitability(role):

    role = str(role).lower()

    for r in GOOD_ROLES:

        if r in role:
            return 1.0

    for r in BAD_ROLES:

        if r in role:
            return 0.0

    return 0.50


# ==========================================================
# Resume Statistics
# ==========================================================

def lexical_diversity(text):

    words = text.lower().split()

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)


def resume_length_score(text):

    words = len(text.split())

    if words >= 500:
        return 1.0

    return min(words / 500, 1.0)


# ==========================================================
# Skill Features
# ==========================================================

def skill_density(skill_list, resume_text):

    words = len(resume_text.split())

    if words == 0:
        return 0

    return len(skill_list) / words


def experience_per_skill(skill_list):

    if len(skill_list) == 0:
        return 0

    months = [

        s["months"]

        for s in skill_list

    ]

    return np.mean(months)


def endorsement_density(skill_list):

    if len(skill_list) == 0:
        return 0

    endorsements = [

        s["endorsements"]

        for s in skill_list

    ]

    return np.mean(endorsements)


# ==========================================================
# Experience Features
# ==========================================================

def experience_bonus(months):

    years = months / 12

    if years >= 8:
        return 1.0

    if years >= 6:
        return 0.9

    if years >= 4:
        return 0.8

    if years >= 2:
        return 0.65

    return 0.40


def seniority_level(months):

    years = months / 12

    if years >= 10:
        return "Principal"

    if years >= 7:
        return "Senior"

    if years >= 4:
        return "Mid-Senior"

    if years >= 2:
        return "Mid-Level"

    return "Junior"


# ==========================================================
# Career Stability
# ==========================================================

def career_stability(role, months):

    score = 0.5

    role = str(role).lower()

    if "senior" in role:

        score += 0.20

    if "lead" in role:

        score += 0.20

    if "principal" in role:

        score += 0.20

    if months >= 48:

        score += 0.10

    return min(score, 1.0)


# ==========================================================
# Derived Candidate Features
# ==========================================================

def build_candidate_features(candidate):

    skill_set = candidate_skill_set(
        candidate["skills"]
    )

    resume = candidate["resume_text"]

    role = candidate["career"]["role"]

    months = candidate["career"]["months"]

    return {

        "role_score":
            role_suitability(role),

        "lexical_diversity":
            lexical_diversity(resume),

        "resume_length":
            resume_length_score(resume),

        "skill_density":
            skill_density(
                candidate["skills"],
                resume
            ),

        "experience_per_skill":
            experience_per_skill(
                candidate["skills"]
            ),

        "endorsement_density":
            endorsement_density(
                candidate["skills"]
            ),

        "experience_bonus":
            experience_bonus(months),

        "career_stability":
            career_stability(
                role,
                months
            ),

        "seniority":
            seniority_level(months)

    }

# ==========================================================
# Career Evidence Integration
# ==========================================================

from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = [

    "career_evidence_score",

    "skill_strength",

    "experience_score",

    "education_strength",

    "role_score",

    "semantic_alignment",

    "semantic_score",

    "retrieval_score",

    "rrf_score",

    "matched_skill_ratio",

    "important_skill_score",

    "keyword_density",

    "resume_completeness",

    "career_stability",

    "resume_length",

    "lexical_diversity",

    "endorsement_density",

    "skill_density",

    "experience_bonus",

    "experience_per_skill"

]


# ==========================================================
# Safe Normalization
# ==========================================================

def normalize_features(df):

    scaler = MinMaxScaler()

    for col in FEATURE_COLUMNS:

        if col not in df.columns:
            continue

        values = df[[col]].fillna(0)

        if values.nunique().iloc[0] <= 1:

            df[col] = values

        else:

            df[col] = scaler.fit_transform(values)

    return df


# ==========================================================
# Build JD Features
# ==========================================================

def build_jd_features(df,
                      jd_keywords):

    matched = []

    ratios = []

    keyword_hits_list = []

    keyword_density_list = []

    important_scores = []

    completeness = []

    for _, row in df.iterrows():

        skills = candidate_skill_set(
            row["parsed_skills"]
        )

        resume = row["resume_text"]

        matched.append(
            jd_skill_overlap(
                skills,
                jd_keywords
            )
        )

        ratios.append(
            matched_skill_ratio(
                skills,
                jd_keywords
            )
        )

        keyword_hits_list.append(

            keyword_hits(
                resume,
                jd_keywords
            )

        )

        keyword_density_list.append(

            keyword_density(
                resume,
                jd_keywords
            )

        )

        important_scores.append(

            important_skill_score(
                skills
            )

        )

        completeness.append(

            resume_completeness(

                {

                    "career":
                        row["career"],

                    "skills":
                        row["parsed_skills"],

                    "education":
                        row["education"],

                    "certifications":
                        row["certifications"]

                }

            )

        )

    df["matched_skills"] = matched

    df["matched_skill_ratio"] = ratios

    df["keyword_hits"] = keyword_hits_list

    df["keyword_density"] = keyword_density_list

    df["important_skill_score"] = important_scores

    df["resume_completeness"] = completeness

    return df


# ==========================================================
# Candidate Derived Features
# ==========================================================

def build_candidate_statistics(df):

    derived = []

    for _, row in df.iterrows():

        candidate = {

            "resume_text":
                row["resume_text"],

            "skills":
                row["parsed_skills"],

            "career":
                row["career"]

        }

        derived.append(

            build_candidate_features(
                candidate
            )

        )

    derived = pd.DataFrame(derived)

    return pd.concat(

        [

            df.reset_index(drop=True),

            derived

        ],

        axis=1

    )

    candidate_df = build_features(
        candidate_df,
        jd_text
    )

# ==========================================================
# Semantic Feature Integration
# ==========================================================

def integrate_semantic_scores(
    df,
    semantic_alignment,
    semantic_scores=None,
    retrieval_scores=None,
    rrf_scores=None
):

    df = df.copy()

    df["semantic_alignment"] = semantic_alignment

    if semantic_scores is not None:
        df["semantic_score"] = semantic_scores

    if retrieval_scores is not None:
        df["retrieval_score"] = retrieval_scores

    if rrf_scores is not None:
        df["rrf_score"] = rrf_scores

    return df


# ==========================================================
# Career Engine Integration
# ==========================================================

from .career_engine import career_evidence


def integrate_career_engine(df):

    career_features = []

    for _, row in df.iterrows():

        candidate = {

            "career": row["career"],

            "skills": row["parsed_skills"],

            "education": row["education"],

            "certifications": row["certifications"]

        }

        career_features.append(

            career_evidence(candidate)

        )

    career_df = pd.DataFrame(career_features)

    df = pd.concat(

        [

            df.reset_index(drop=True),

            career_df

        ],

        axis=1

    )

    return df


# ==========================================================
# Final Feature Builder
# ==========================================================

def build_features(

    candidate_df,

    jd_text,

    semantic_alignment,

    semantic_scores=None,

    retrieval_scores=None,

    rrf_scores=None

):

    print("Building JD Keywords...")

    jd_keywords = extract_keywords(jd_text)

    print("Building JD Features...")

    candidate_df = build_jd_features(

        candidate_df,

        jd_keywords

    )

    print("Building Candidate Features...")

    candidate_df = build_candidate_statistics(

        candidate_df

    )

    print("Integrating Career Engine...")

    candidate_df = integrate_career_engine(

        candidate_df

    )

    print("Adding Semantic Features...")

    candidate_df = integrate_semantic_scores(

        candidate_df,

        semantic_alignment,

        semantic_scores,

        retrieval_scores,

        rrf_scores

    )

    print("Normalizing Features...")

    candidate_df = normalize_features(

        candidate_df

    )

    return candidate_df


# ==========================================================
# Export Features
# ==========================================================

def export_feature_matrix(df):

    cols = [

        c

        for c in FEATURE_COLUMNS

        if c in df.columns

    ]

    cols = [

        "candidate_id"

    ] + cols

    return df[cols]


