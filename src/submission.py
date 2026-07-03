"""
submission.py

Creates the final Redrob AI Hiring Challenge submission.

Author: Harshit Goyal
"""

import pandas as pd


REQUIRED_COLUMNS = [

    "candidate_id",

    "rank",

    "score",

    "reasoning"

]


# ==========================================================
# Validation
# ==========================================================

def validate_submission(df):

    missing = [

        c

        for c in REQUIRED_COLUMNS

        if c not in df.columns

    ]

    if len(missing):

        raise ValueError(

            f"Missing columns: {missing}"

        )

    if df["candidate_id"].duplicated().any():

        raise ValueError(

            "Duplicate candidate IDs found."

        )

    if df["rank"].duplicated().any():

        raise ValueError(

            "Duplicate ranks found."

        )

    return True


# ==========================================================
# Score Scaling
# ==========================================================

def prepare_scores(df):

    df = df.copy()

    if "ranking_score" in df.columns:

        df["score"] = (

            df["ranking_score"]

            / 100

        ).round(3)

    elif "raw_score" in df.columns:

        mn = df["raw_score"].min()

        mx = df["raw_score"].max()

        df["score"] = (

            0.70 +

            (df["raw_score"]-mn)

            *

            0.29

            /

            (mx-mn)

        ).round(3)

    return df

# ==========================================================
# Build Submission
# ==========================================================

def build_submission(

    ranked_df,

    top_k=100

):

    submission = (

        ranked_df

        .head(top_k)

        .copy()

    )

    submission = prepare_scores(

        submission

    )

    submission = submission[

        [

            "candidate_id",

            "rank",

            "score",

            "reasoning"

        ]

    ]

    validate_submission(

        submission

    )

    return submission

# ==========================================================
# Save CSV
# ==========================================================

def save_submission(

    submission,

    filename="submission.csv"

):

    submission.to_csv(

        filename,

        index=False

    )

    print(

        f"Submission saved -> {filename}"

    )

# ==========================================================
# Recruiter Report
# ==========================================================

def save_recruiter_report(

    ranked_df,

    filename="recruiter_report.csv",

    top_k=100

):

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

        if c in ranked_df.columns

    ]

    ranked_df.head(

        top_k

    )[cols].to_csv(

        filename,

        index=False

    )

    print(

        f"Recruiter report saved -> {filename}"

    )

# ==========================================================
# Save Full Ranking
# ==========================================================

def save_full_ranking(

    ranked_df,

    filename="final_candidate_ranking.parquet"

):

    ranked_df.to_parquet(

        filename,

        index=False

    )

    print(

        f"Ranking saved -> {filename}"

    )


# ==========================================================
# Export All
# ==========================================================

def export_all(

    ranked_df,

    top_k=100

):

    submission = build_submission(

        ranked_df,

        top_k

    )

    save_submission(

        submission

    )

    save_recruiter_report(

        ranked_df,

        top_k=top_k

    )

    save_full_ranking(

        ranked_df

    )

    return submission

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Redrob AI Hiring Submission Module")
    print("=" * 60)

    print(
        "Import export_all() from this module to generate the final outputs."
    )

