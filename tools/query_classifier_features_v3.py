"""Serializable query-only feature transformers for the V3 classifier."""

from __future__ import annotations

import re
from typing import Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from collections import Counter


class QueryTextSelector(BaseEstimator, TransformerMixin):
    """Raw query text for the two TF-IDF blocks, without structural pseudo tokens."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [str(item.get('question') or '') if isinstance(item, dict) else str(item or '') for item in X]


class QueryStructureFeatures(BaseEstimator, TransformerMixin):
    """Independent sparse structural block; vocabulary is fitted on training rows only."""

    def __init__(self, cue_repeats=1):
        self.cue_repeats = cue_repeats

    def records(self, X):
        selector = CueAugmentedTextSelector(self.cue_repeats)
        return [dict(Counter(selector.cues(text))) for text in QueryTextSelector().transform(X)]

    def fit(self, X, y=None):
        self.vectorizer_ = DictVectorizer(sparse=True).fit(self.records(X))
        return self

    def transform(self, X):
        return self.vectorizer_.transform(self.records(X))

    def fit_transform(self, X, y=None, **fit_params):
        rows = list(X)
        return self.fit(rows, y).transform(rows)


class CueAugmentedTextSelector(BaseEstimator, TransformerMixin):
    """Convert query-visible structural cues into sparse pseudo tokens."""

    patterns = {
        "compare": re.compile(
            r"\b(older|younger|earlier|later|higher|lower|larger|smaller|"
            r"both|same|different|respectively|before|after)\b"
        ),
        "explain": re.compile(
            r"(^why\b|^how (does|do|did|can|is|are)\b|"
            r"\b(explain|meaning|cause|purpose|significance|impact|effect|outcome)\b)"
        ),
        "bridge": re.compile(r"\b(who|that|which|whose|where)\b"),
        "relative": re.compile(
            r"\b(the (city|person|film|song|author|director|player|team|country|"
            r"company|university|school|river|show|book) (who|that|which|where|whose)|"
            r"the (director|author|spouse|father|mother|capital|birthplace) of)\b"
        ),
        "quantity": re.compile(r"^how (many|much|large|long|old)\b"),
        "choice": re.compile(r"^which\b"),
    }

    def __init__(self, cue_repeats: int = 2):
        self.cue_repeats = cue_repeats

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[dict]) -> List[str]:
        output = []
        for item in X:
            original = str(item.get("question") or "") if isinstance(item, dict) else str(item or "")
            output.append(f"{original.lower().strip()} {' '.join(self.cues(original))}")
        return output

    def cues(self, original):
        question = original.lower().strip()
        words = re.findall(r"[a-z0-9][a-z0-9'_-]*", question)
        first_word = words[0] if words else 'empty'
        cues = [*([f'__wh_{first_word}'] * 2), f'__len_{min(len(words) // 4, 7)}',
                f"__of_{min(question.count(' of '), 3)}", f"__comma_{min(question.count(','), 2)}"]
        for name, pattern in self.patterns.items():
            if pattern.search(question):
                cues.extend([f'__{name}'] * self.cue_repeats)
        if len(re.findall(r'\b[A-Z][a-z]+\b', original)) >= 2:
            cues.extend(['__multi_cap'] * self.cue_repeats)
        return cues
