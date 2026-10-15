"""Small, inspectable TF-IDF RAG implementation."""
from __future__ import annotations
import re
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity
from ingestion import load_documents

class RajasthanRetriever:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else load_documents()
        # Word features preserve interpretable phrase matches; character features
        # make mixed Hindi/English spellings (including common Romanized Hindi)
        # less brittle without introducing a heavyweight embedding dependency.
        word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        self.vectorizer = FeatureUnion([
            ("words", word_vectorizer),
            ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1, sublinear_tf=True)),
        ])
        corpus = [f"{d['title']} {d['destination']} {d['category']} {d['text']}" for d in self.documents]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None
        self.word_matrix = word_vectorizer.transform(corpus) if corpus else None

    def search(self, query, destination=None, category=None, top_k=5):
        if not query or not query.strip() or self.matrix is None: return []
        candidates = [(i, d) for i, d in enumerate(self.documents)
                      if (not destination or d["destination"].lower() == destination.lower())
                      and (not category or d["category"].lower() == category.lower())]
        if not candidates: return []
        # The destination is already a hard filter. Leaving it in the query makes
        # every row in the filtered set score highly on the shared city name, which
        # drowns out the actually discriminative terms ("bus", "safari", "food").
        # Drop those tokens so ranking reflects what the traveler actually asked.
        q_text = self._strip_filter_terms(query, destination)
        if not q_text.strip(): return []
        q = self.vectorizer.transform([q_text])
        indices = [i for i, _ in candidates]
        scores = cosine_similarity(q, self.matrix[indices]).ravel()
        # Keep character features from manufacturing relevance via short accidental
        # overlaps: a candidate must also share at least one meaningful word/phrase.
        word_scores = cosine_similarity(
            self.vectorizer.transformer_list[0][1].transform([q_text]), self.word_matrix[indices]).ravel()
        rows = []
        for score, word_score, (_, doc) in sorted(zip(scores, word_scores, candidates), reverse=True, key=lambda x: x[0]):
            if score >= 0.03 and word_score > 0:
                rows.append({**doc, "score": round(float(score), 3)})
        return rows[:max(1, min(top_k, 10))]

    @staticmethod
    def _strip_filter_terms(query, destination):
        """Remove destination-name tokens that the row filter has already applied."""
        if not destination:
            return query
        blocked = {t for t in re.findall(r"\w+", destination.lower()) if len(t) > 1}
        if not blocked:
            return query
        kept = [t for t in re.findall(r"[\wऀ-ॿ]+", query) if t.lower() not in blocked]
        return " ".join(kept) or query

@lru_cache(maxsize=1)
def get_retriever(): return RajasthanRetriever()

def retrieve_knowledge(query, destination=None, category=None, top_k=5):
    return get_retriever().search(query, destination, category, top_k)
