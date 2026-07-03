"""
utils.py

Common utility functions used across the Redrob AI Hiring pipeline.
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Text Utilities
# ==========================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    return text.strip()


def normalize_whitespace(text: str) -> str:

    if pd.isna(text):
        return ""

    return " ".join(str(text).split())


def word_count(text: str) -> int:

    return len(clean_text(text).split())


def char_count(text: str) -> int:

    return len(str(text))


# ==========================================================
# Safe Parsing
# ==========================================================

def safe_json_loads(x):

    if isinstance(x, dict):
        return x

    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    try:
        return json.loads(x)

    except Exception:
        return []


def safe_eval_list(x):

    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    try:
        import ast
        return ast.literal_eval(x)

    except Exception:
        return []


# ==========================================================
# Scaling
# ==========================================================

def minmax(series):

    scaler = MinMaxScaler()

    values = scaler.fit_transform(
        np.array(series).reshape(-1,1)
    )

    return values.flatten()


def normalize_column(df, column):

    df[column] = minmax(df[column])

    return df


# ==========================================================
# Skill Utilities
# ==========================================================

def clean_skill(skill):

    if pd.isna(skill):
        return ""

    skill = str(skill)

    skill = skill.lower()

    skill = skill.replace("_"," ")

    skill = re.sub(r"\s+"," ",skill)

    return skill.strip()


def unique_skills(skill_list):

    cleaned = []

    seen = set()

    for skill in skill_list:

        skill = clean_skill(skill)

        if skill not in seen:

            cleaned.append(skill)

            seen.add(skill)

    return cleaned


# ==========================================================
# Keyword Matching
# ==========================================================

def keyword_overlap(text, keywords):

    text = clean_text(text)

    count = 0

    for kw in keywords:

        if kw.lower() in text:

            count += 1

    return count


def keyword_ratio(text, keywords):

    if len(keywords)==0:
        return 0

    return keyword_overlap(text,keywords)/len(keywords)


# ==========================================================
# Cosine Helpers
# ==========================================================

def cosine_to_score(similarity):

    """
    Convert cosine similarity into
    recruiter-friendly score.
    """

    similarity = max(0,min(1,similarity))

    return round(similarity*100,2)


# ==========================================================
# Experience Utilities
# ==========================================================

def months_to_years(months):

    return round(months/12,1)


def experience_level(months):

    years = months/12

    if years>=10:
        return "Principal"

    if years>=7:
        return "Senior"

    if years>=4:
        return "Mid-Senior"

    if years>=2:
        return "Mid-Level"

    return "Junior"


# ==========================================================
# Ranking Helpers
# ==========================================================

def assign_rank(df, score_column):

    df = df.sort_values(
        score_column,
        ascending=False
    ).reset_index(drop=True)

    df["rank"] = np.arange(
        1,
        len(df)+1
    )

    return df


# ==========================================================
# Display
# ==========================================================

def print_header(title):

    print("="*60)

    print(title)

    print("="*60)