"""
career_engine.py

Career Evidence Engine

Responsible for converting parsed candidate
information into recruiter-oriented career signals.
"""

import numpy as np
import pandas as pd


# ==========================================================
# Skill Strength
# ==========================================================

PROFICIENCY_SCORE = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4
}


def compute_skill_strength(skills):

    if len(skills) == 0:
        return 0

    total = 0

    for skill in skills:

        prof = PROFICIENCY_SCORE.get(
            skill["proficiency"].lower(),
            1
        )

        months = skill["months"] / 12

        endorse = np.log1p(skill["endorsements"])

        total += prof * (1 + months) * (1 + endorse)

    return total / len(skills)


# ==========================================================
# Experience Score
# ==========================================================

def experience_score(months):

    years = months / 12

    if years >= 10:
        return 1.0

    if years >= 7:
        return 0.90

    if years >= 5:
        return 0.80

    if years >= 3:
        return 0.70

    if years >= 2:
        return 0.60

    if years >= 1:
        return 0.45

    return 0.25


# ==========================================================
# Education Score
# ==========================================================

DEGREE_SCORE = {

    "PHD": 1.00,

    "M.TECH": 0.90,
    "M.E.": 0.90,
    "M.SC": 0.85,

    "B.TECH": 0.80,
    "B.E.": 0.80,

    "B.SC": 0.70,

    "MBA": 0.60
}


def education_score(education):

    degree = education.get("degree", "").upper()

    cgpa = education.get("cgpa", 0)

    tier = education.get("tier", 4)

    score = DEGREE_SCORE.get(degree, 0.50)

    if cgpa:

        score += (cgpa / 10) * 0.10

    if tier:

        score += (5 - tier) * 0.03

    return min(score, 1.0)


# ==========================================================
# Certification Score
# ==========================================================

def certification_score(certs):

    n = len(certs)

    if n == 0:
        return 0

    return min(1.0, 0.20 * n)


# ==========================================================
# Career Consistency
# ==========================================================

GOOD_ROLES = {

    "machine learning engineer",

    "ml engineer",

    "ai engineer",

    "nlp engineer",

    "search engineer",

    "recommendation systems engineer",

    "data scientist",

    "backend engineer",

    "software engineer",

    "applied ml engineer"

}


def role_consistency(role):

    role = str(role).lower()

    for good in GOOD_ROLES:

        if good in role:
            return 1.0

    return 0.50


# ==========================================================
# Career Evidence Score
# ==========================================================

def career_evidence(candidate):

    skill_score = compute_skill_strength(
        candidate["skills"]
    )

    exp_score = experience_score(
        candidate["career"]["months"]
    )

    edu_score = education_score(
        candidate["education"]
    )

    cert_score = certification_score(
        candidate["certifications"]
    )

    role_score = role_consistency(
        candidate["career"]["role"]
    )

    final = (

        0.35 * skill_score +

        0.25 * exp_score +

        0.15 * edu_score +

        0.10 * cert_score +

        0.15 * role_score

    )

    return {

        "career_evidence_score": final,

        "skill_strength": skill_score,

        "experience_score": exp_score,

        "education_strength": edu_score,

        "certification_strength": cert_score,

        "role_score": role_score

    }


# ==========================================================
# Batch Processing
# ==========================================================

def build_career_features(parsed_candidates):

    features = []

    for candidate in parsed_candidates:

        features.append(
            career_evidence(candidate)
        )

    return pd.DataFrame(features)