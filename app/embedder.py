"""Embedding layer.

Primary (synopsis): Sentence-Transformers all-MiniLM-L6-v2.
Deploy backend: fastembed — the SAME MiniLM model in ONNX form, no torch,
small enough for free-tier hosting (Render 512 MB). Embeddings are the
model's own, so scores are comparable with the torch run.
Fallback: TF-IDF (scikit-learn) so the demo runs offline / without either.
The rest of the pipeline does not care which backend produced the vectors —
they are L2-normalised, so FAISS inner-product == cosine similarity.
"""
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, corpus_texts=None):
        self.backend = "tfidf"
        self._model = None
        self._vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
            self.backend = "sbert"
            return
        except Exception:
            pass
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            self.backend = "sbert-onnx"
            return
        except Exception:
            pass
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"
        )
        if corpus_texts:
            self._vectorizer.fit(corpus_texts)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype="float32")
        if v.ndim == 1:
            v = v.reshape(1, -1)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "sbert":
            v = self._model.encode(texts, convert_to_numpy=True)
        elif self.backend == "sbert-onnx":
            v = np.array(list(self._model.embed(texts)), dtype="float32")
        else:
            v = self._vectorizer.transform(texts).toarray()
        return self._norm(v)
