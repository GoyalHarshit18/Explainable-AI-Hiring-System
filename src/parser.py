"""
parser.py

Parses raw candidate documents into structured Python objects.
"""

import re
import pandas as pd


# ==========================================================
# Skill Parser
# ==========================================================

SKILL_PATTERN = re.compile(
    r"The candidate has\s+(.*?)\s+proficiency in\s+(.*?)\s+with\s+(\d+)\s+months of experience\s+and\s+(\d+)\s+endorsements",
    re.IGNORECASE
)


def parse_skills(text):

    if pd.isna(text):
        return []

    skills = []

    for match in SKILL_PATTERN.finditer(str(text)):

        proficiency = match.group(1).strip().lower()
        skill = match.group(2).strip().lower()
        months = int(match.group(3))
        endorsements = int(match.group(4))

        skills.append({

            "skill": skill,
            "proficiency": proficiency,
            "months": months,
            "endorsements": endorsements

        })

    return skills


# ==========================================================
# Education Parser
# ==========================================================

DEGREE_PATTERN = re.compile(
    r"Completed\s+(.*?)\s+in\s+(.*?)\s+from\s+(.*?)\.",
    re.IGNORECASE
)

CGPA_PATTERN = re.compile(
    r"Academic grade:\s*([0-9.]+)",
    re.IGNORECASE
)

TIER_PATTERN = re.compile(
    r"Institution tier:\s*tier[_ ]?(\d)",
    re.IGNORECASE
)


def parse_education(text):

    if pd.isna(text):
        return {}

    text = str(text)

    degree = ""
    domain = ""
    institute = ""

    m = DEGREE_PATTERN.search(text)

    if m:
        degree = m.group(1).strip()
        domain = m.group(2).strip()
        institute = m.group(3).strip()

    cgpa = None

    m = CGPA_PATTERN.search(text)

    if m:
        cgpa = float(m.group(1))

    tier = None

    m = TIER_PATTERN.search(text)

    if m:
        tier = int(m.group(1))

    return {

        "degree": degree,
        "domain": domain,
        "institute": institute,
        "cgpa": cgpa,
        "tier": tier

    }


# ==========================================================
# Career Parser
# ==========================================================

ROLE_PATTERN = re.compile(
    r"currently works\s+as\s+a\s+(.*?)\s+at",
    re.IGNORECASE
)

COMPANY_PATTERN = re.compile(
    r"at\s+(.*?)\s+for",
    re.IGNORECASE
)

EXP_PATTERN = re.compile(
    r"for\s+(\d+)\s+months",
    re.IGNORECASE
)

INDUSTRY_PATTERN = re.compile(
    r"industry:\s*(.*?)\.",
    re.IGNORECASE
)


def parse_career(text):

    if pd.isna(text):
        return {}

    text = str(text)

    role = ""
    company = ""
    months = 0
    industry = ""

    m = ROLE_PATTERN.search(text)

    if m:
        role = m.group(1).strip().lower()

    m = COMPANY_PATTERN.search(text)

    if m:
        company = m.group(1).strip()

    m = EXP_PATTERN.search(text)

    if m:
        months = int(m.group(1))

    m = INDUSTRY_PATTERN.search(text)

    if m:
        industry = m.group(1).strip()

    return {

        "role": role,
        "company": company,
        "months": months,
        "industry": industry

    }


# ==========================================================
# Certifications
# ==========================================================

def parse_certifications(text):

    if pd.isna(text):
        return []

    text = str(text)

    certs = []

    for line in text.split("\n"):

        line = line.strip()

        if len(line) > 2:
            certs.append(line)

    return certs


# ==========================================================
# Resume Builder
# ==========================================================

def build_resume(row):

    docs = [

        row.get("profile_document",""),

        row.get("career_document",""),

        row.get("skills_document",""),

        row.get("education_document",""),

        row.get("certification_document","")

    ]

    docs = [

        str(x)

        for x in docs

        if pd.notna(x)

    ]

    return "\n".join(docs)


# ==========================================================
# Main Parser
# ==========================================================

def parse_candidate(row):

    return {

        "resume_text": build_resume(row),

        "career": parse_career(
            row.get("career_document","")
        ),

        "skills": parse_skills(
            row.get("skills_document","")
        ),

        "education": parse_education(
            row.get("education_document","")
        ),

        "certifications": parse_certifications(
            row.get("certification_document","")
        )

    }