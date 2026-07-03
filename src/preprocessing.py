"""
preprocessing.py

Resume parsing and preprocessing utilities.
"""

import re
import pandas as pd

from .utils import (
    clean_text,
    word_count,
    char_count,
    normalize_whitespace
)


# ==========================================================
# Resume Builder
# ==========================================================

def build_resume_text(row):

    docs = [

        row.get("profile_document", ""),

        row.get("career_document", ""),

        row.get("skills_document", ""),

        row.get("education_document", ""),

        row.get("certification_document", "")

    ]

    docs = [normalize_whitespace(x) for x in docs]

    return "\n".join(docs)


# ==========================================================
# Current Role
# ==========================================================

ROLE_PATTERN = re.compile(
    r"currently works\s+as\s+a\s+(.*?)\s+at",
    re.IGNORECASE
)


def extract_current_role(text):

    text = normalize_whitespace(text)

    match = ROLE_PATTERN.search(text)

    if match:
        return match.group(1).strip().lower()

    return ""


# ==========================================================
# Experience
# ==========================================================

MONTH_PATTERN = re.compile(
    r"(\d+)\s+months",
    re.IGNORECASE
)


def extract_total_experience(text):

    months = MONTH_PATTERN.findall(text)

    if len(months) == 0:
        return 0

    months = [int(x) for x in months]

    return max(months)


# ==========================================================
# Education
# ==========================================================

DEGREE_PATTERN = re.compile(
    r"(b\.tech|b\.e\.|b\.sc|m\.tech|m\.e\.|m\.sc|phd|mba)",
    re.IGNORECASE
)


def extract_degree(text):

    match = DEGREE_PATTERN.search(text)

    if match:
        return match.group(1).upper()

    return ""


CGPA_PATTERN = re.compile(
    r"(\d+\.\d+)\s*cgpa",
    re.IGNORECASE
)


def extract_cgpa(text):

    match = CGPA_PATTERN.search(text)

    if match:
        return float(match.group(1))

    return None


TIER_PATTERN = re.compile(
    r"tier[_ ]?(\d)",
    re.IGNORECASE
)


def extract_institute_tier(text):

    match = TIER_PATTERN.search(text)

    if match:
        return int(match.group(1))

    return None


# ==========================================================
# Certifications
# ==========================================================

def certification_count(text):

    if pd.isna(text):
        return 0

    text = clean_text(text)

    if len(text) < 5:
        return 0

    return len(
        [
            x
            for x in text.split(".")
            if len(x.strip()) > 2
        ]
    )


# ==========================================================
# Resume Statistics
# ==========================================================

def compute_resume_statistics(df):

    df["resume_text"] = df.apply(
        build_resume_text,
        axis=1
    )

    df["word_count"] = (
        df["resume_text"]
        .apply(word_count)
    )

    df["char_count"] = (
        df["resume_text"]
        .apply(char_count)
    )

    return df


# ==========================================================
# Career Parsing
# ==========================================================

def parse_career(df):

    df["current_role"] = (
        df["career_document"]
        .apply(extract_current_role)
    )

    df["career_months"] = (
        df["career_document"]
        .apply(extract_total_experience)
    )

    return df


# ==========================================================
# Education Parsing
# ==========================================================

def parse_education(df):

    df["degree"] = (
        df["education_document"]
        .apply(extract_degree)
    )

    df["cgpa"] = (
        df["education_document"]
        .apply(extract_cgpa)
    )

    df["institution_tier"] = (
        df["education_document"]
        .apply(extract_institute_tier)
    )

    return df


# ==========================================================
# Certification Parsing
# ==========================================================

def parse_certifications(df):

    df["num_certifications"] = (
        df["certification_document"]
        .apply(certification_count)
    )

    return df


# ==========================================================
# Complete Pipeline
# ==========================================================

def preprocess_candidates(df):

    df = compute_resume_statistics(df)

    df = parse_career(df)

    df = parse_education(df)

    df = parse_certifications(df)

    return df