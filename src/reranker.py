"""
reranker.py

Main AI Hiring Pipeline

End-to-end pipeline for
Hybrid Retrieval
→ Feature Engineering
→ Career Evidence
→ Scoring
→ Dynamic Reasoning
"""

import pandas as pd

from .preprocessing import preprocess_candidates
from .feature_engineering import build_features
from .scoring import score_candidates
from .reasoning import build_reasoning


class AIReranker:

    """
    End-to-end AI hiring reranker.
    """

    def __init__(self):

        pass

    def run(

        self,

        candidate_df,

        jd_text,

        semantic_alignment,

        semantic_scores=None,

        retrieval_scores=None,

        rrf_scores=None

    ):

        print("=" * 60)
        print("AI Hiring Pipeline Started")
        print("=" * 60)

        # -----------------------------------------
        # Preprocessing
        # -----------------------------------------

        print("Step 1 : Preprocessing...")

        candidate_df = preprocess_candidates(

            candidate_df

        )

        # -----------------------------------------
        # Feature Engineering
        # -----------------------------------------

        print("Step 2 : Feature Engineering...")

        candidate_df = build_features(

            candidate_df,

            jd_text,

            semantic_alignment,

            semantic_scores,

            retrieval_scores,

            rrf_scores

        )

        # -----------------------------------------
        # Final Scoring
        # -----------------------------------------

        print("Step 3 : Candidate Ranking...")

        candidate_df = score_candidates(

            candidate_df

        )

        # -----------------------------------------
        # Explainability
        # -----------------------------------------

        print("Step 4 : Dynamic Recruiter Reasoning...")

        candidate_df = build_reasoning(

            candidate_df

        )

        print("=" * 60)
        print("Pipeline Finished Successfully")
        print("=" * 60)

        return candidate_df
    

# ==========================================================
# Top Candidates
# ==========================================================

def top_candidates(

    candidate_df,

    k=100

):

    return (

        candidate_df

        .sort_values(

            "raw_score",

            ascending=False

        )

        .head(k)

        .reset_index(drop=True)

    )


# ==========================================================
# Recruiter Report
# ==========================================================

def recruiter_report(

    candidate_df,

    k=100

):

    report = top_candidates(

        candidate_df,

        k

    )

    cols = [

        "rank",

        "candidate_id",

        "current_role",

        "career_months",

        "ranking_score",

        "reasoning"

    ]

    cols = [

        c

        for c in cols

        if c in report.columns

    ]

    return report[cols]


# ==========================================================
# Feature Matrix
# ==========================================================

def feature_matrix(candidate_df):

    ignore = {

        "resume_text",

        "career",

        "parsed_skills",

        "education",

        "certifications"

    }

    cols = [

        c

        for c in candidate_df.columns

        if c not in ignore

    ]

    return candidate_df[cols]

if __name__ == "__main__":

    print(
        "AIReranker module loaded successfully."
    )