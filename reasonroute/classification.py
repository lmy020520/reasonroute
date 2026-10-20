"""Query-only feature configuration and classifier metrics."""
from typing import Dict, Iterable, List, Sequence, Tuple
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tools.query_classifier_features_v3 import QueryTextSelector, QueryStructureFeatures
FEATURE_SCHEMA = 'paper-three-block-v1'
LABELS = ["fact", "explanation", "reasoning"]


def build_model(
    *,
    cue_repeats: int,
    word_ngram: Tuple[int, int],
    char_ngram: Tuple[int, int],
    min_df: int,
    c_value: float,
) -> Pipeline:
    selector = QueryTextSelector()
    features = FeatureUnion(
        [
            (
                "word",
                Pipeline(
                    [
                        ("text", selector),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=word_ngram,
                                min_df=min_df,
                                sublinear_tf=True,
                                strip_accents="unicode",
                            ),
                        ),
                    ]
                ),
            ),
            (
                "char",
                Pipeline(
                    [
                        ("text", selector),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=char_ngram,
                                min_df=min_df,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ('structure', QueryStructureFeatures(cue_repeats=cue_repeats)),
        ],
        transformer_weights={"word": 1.25, "char": 0.75, "structure": 1.0},
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    loss="hinge",
                    max_iter=15000,
                ),
            ),
        ]
    )

def metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted"),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
