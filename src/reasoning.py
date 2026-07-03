"""
reasoning.py

Dynamic Explainable AI Reasoning Engine
"""

import pandas as pd


# ==========================================================
# Skill Categories
# ==========================================================

SKILL_CATEGORIES = {

    "Retrieval & Search":[
        "faiss","bm25","elasticsearch","retrieval",
        "vector search","pinecone","milvus",
        "qdrant","weaviate"
    ],

    "NLP & LLM":[
        "nlp","bert","transformers",
        "sentence transformers",
        "langchain",
        "llm",
        "huggingface",
        "spacy"
    ],

    "Machine Learning":[
        "tensorflow",
        "pytorch",
        "xgboost",
        "lightgbm",
        "catboost",
        "machine learning",
        "scikit-learn"
    ],

    "Recommendation Systems":[
        "recommendation",
        "ranking",
        "recommender",
        "collaborative filtering"
    ],

    "Cloud & Deployment":[
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "mlflow"
    ]
}

# ==========================================================
# Detect Candidate Domain
# ==========================================================

def detect_specialization(skills):

    skills = [str(s).lower() for s in skills]

    score = {}

    for category, keywords in SKILL_CATEGORIES.items():

        hits = 0

        for kw in keywords:

            for skill in skills:

                if kw in skill:

                    hits += 1

        score[category] = hits

    best = max(score, key=score.get)

    if score[best] == 0:

        return "Software Engineering"

    return best

# ==========================================================
# Recruiter Explanation
# ==========================================================

SPECIALIZATION_REASON = {

    "Retrieval & Search":
        "demonstrates expertise in search, retrieval and vector databases",

    "NLP & LLM":
        "has strong experience in NLP and modern language models",

    "Machine Learning":
        "shows solid machine learning and model development experience",

    "Recommendation Systems":
        "has practical recommendation and ranking system experience",

    "Cloud & Deployment":
        "has experience deploying production AI systems",

    "Software Engineering":
        "has a strong software engineering background"
}

# ==========================================================
# Main Reason Generator
# ==========================================================

def generate_reason(row):

    years = round(

        row["career_months"]/12,

        1

    )

    specialization = detect_specialization(

        row["top_skills"]

    )

    explanation = SPECIALIZATION_REASON.get(

        specialization,

        "has a relevant technical background"

    )

    top_skills = ", ".join(

        row["top_skills"][:3]

    )

    return (

        f"{row['current_role'].title()} "

        f"with {years} years of experience; "

        f"{explanation}. "

        f"Strong expertise in {top_skills}."

    )

# ==========================================================
# Build Column
# ==========================================================

def build_reasoning(df):

    df = df.copy()

    df["reasoning"] = (

        df

        .apply(

            generate_reason,

            axis=1

        )

    )

    return df

# ==========================================================
# Preview
# ==========================================================

def preview_reasoning(df,n=5):

    return df[
        [

            "candidate_id",

            "reasoning"

        ]

    ].head(n)