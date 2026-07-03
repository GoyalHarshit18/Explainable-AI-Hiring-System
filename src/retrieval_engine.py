"""
retrieval.py

Hybrid Retrieval Engine
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# Semantic Retrieval
# ==========================================================

class SemanticRetriever:

    def __init__(self,
                 candidate_embeddings,
                 candidate_ids):

        self.embeddings = candidate_embeddings
        self.ids = candidate_ids

    def search(self,
               query_embedding,
               top_k=3000):

        similarity = cosine_similarity(
            self.embeddings,
            query_embedding.reshape(1,-1)
        ).flatten()

        idx = np.argsort(
            similarity
        )[::-1][:top_k]

        return pd.DataFrame({

            "candidate_id":
                self.ids[idx],

            "semantic_score":
                similarity[idx]

        })
    
# ==========================================================
# Reciprocal Rank Fusion
# ==========================================================

def reciprocal_rank_fusion(
        semantic_df,
        lexical_df,
        k=60):

    semantic_df = semantic_df.copy()

    lexical_df = lexical_df.copy()

    semantic_df["semantic_rank"] = (
        np.arange(len(semantic_df))
    )

    lexical_df["lexical_rank"] = (
        np.arange(len(lexical_df))
    )

    merged = semantic_df.merge(

        lexical_df,

        on="candidate_id",

        how="outer"

    )

    merged["semantic_rank"] = (
        merged["semantic_rank"]
        .fillna(100000)
    )

    merged["lexical_rank"] = (
        merged["lexical_rank"]
        .fillna(100000)
    )

    merged["rrf_score"] = (

        1 / (k + merged.semantic_rank)

        +

        1 / (k + merged.lexical_rank)

    )

    return (

        merged

        .sort_values(

            "rrf_score",

            ascending=False

        )

        .reset_index(drop=True)

    )

# ==========================================================
# Hybrid Retrieval
# ==========================================================

class HybridRetriever:

    def __init__(

        self,

        semantic_retriever,

        bm25_function

    ):

        self.semantic = semantic_retriever

        self.bm25 = bm25_function

    def retrieve(

        self,

        query_embedding,

        query_text,

        top_k=3000

    ):

        semantic = self.semantic.search(

            query_embedding,

            top_k

        )

        lexical = self.bm25(

            query_text,

            top_k

        )

        hybrid = reciprocal_rank_fusion(

            semantic,

            lexical

        )

        return hybrid.head(top_k)
    

# ==========================================================
# Final Candidate Builder
# ==========================================================

def attach_candidate_metadata(

    retrieved_df,

    metadata

):

    return retrieved_df.merge(

        metadata,

        on="candidate_id",

        how="left"

    )


# ==========================================================
# Retrieval Pipeline
# ==========================================================

def retrieve_candidates(

    query_embedding,

    query_text,

    metadata,

    candidate_embeddings,

    candidate_ids,

    bm25_function,

    top_k=3000

):

    semantic = SemanticRetriever(

        candidate_embeddings,

        candidate_ids

    )

    retriever = HybridRetriever(

        semantic,

        bm25_function

    )

    retrieved = retriever.retrieve(

        query_embedding,

        query_text,

        top_k

    )

    retrieved = attach_candidate_metadata(

        retrieved,

        metadata

    )

    return retrieved