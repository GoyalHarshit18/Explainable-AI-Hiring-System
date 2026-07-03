"""
scoring.py

Final Ranking Engine
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Default Feature Weights
# ==========================================================

DEFAULT_WEIGHTS = {

    "semantic_score":0.20,

    "semantic_alignment":0.20,

    "retrieval_score":0.10,

    "rrf_score":0.05,

    "career_evidence_score":0.10,

    "jd_keyword_feature":0.08,

    "role_score":0.10,

    "skill_strength":0.05,

    "experience_score":0.03,

    "education_strength":0.02,

    "matched_skill_ratio":0.02,

    "important_skill_score":0.02,

    "resume_completeness":0.01,

    "career_stability":0.01,

    "lexical_diversity":0.005,

    "endorsement_density":0.005,

    "skill_density":0.005,

    "experience_bonus":0.005,

    "experience_per_skill":0.005
}


# ==========================================================
# Weighted Score
# ==========================================================

def weighted_score(df,
                   weights=DEFAULT_WEIGHTS):

    df = df.copy()

    df["raw_score"] = 0.0

    for feature, weight in weights.items():

        if feature not in df.columns:
            continue

        df["raw_score"] += (

            df[feature].fillna(0)

            * weight

        )

    return df

# ==========================================================
# Confidence Score
# ==========================================================

def confidence_score(df):

    scaler = MinMaxScaler(
        feature_range=(60,88)
    )

    df = df.copy()

    df["ranking_score"] = scaler.fit_transform(

        df[["raw_score"]]

    )

    df["ranking_score"] = (

        df["ranking_score"]

        .round(2)

    )

    return df

# ==========================================================
# Ranking
# ==========================================================

def rank_candidates(df):

    df = (

        df

        .sort_values(

            "raw_score",

            ascending=False

        )

        .reset_index(drop=True)

    )

    df["rank"] = np.arange(

        1,

        len(df)+1

    )

    return df

# ==========================================================
# Top K
# ==========================================================

def top_candidates(df,
                   k=100):

    return (

        df

        .sort_values(

            "raw_score",

            ascending=False

        )

        .head(k)

        .reset_index(drop=True)

    )

# ==========================================================
# Final Scoring Pipeline
# ==========================================================

def score_candidates(df,
                     weights=DEFAULT_WEIGHTS):

    df = weighted_score(

        df,

        weights

    )

    df = confidence_score(

        df

    )

    df = rank_candidates(

        df

    )

    return df

# ==========================================================
# Feature Contribution
# ==========================================================

def explain_score(row,
                  weights=DEFAULT_WEIGHTS):

    contributions = {}

    for feature, weight in weights.items():

        if feature not in row.index:
            continue

        contributions[feature] = (

            row[feature]

            * weight

        )

    return dict(

        sorted(

            contributions.items(),

            key=lambda x:x[1],

            reverse=True

        )

    )

# ==========================================================
# Export
# ==========================================================

def export_scores(df):

    cols = [

        "candidate_id",

        "rank",

        "raw_score",

        "ranking_score"

    ]

    return df[cols]

