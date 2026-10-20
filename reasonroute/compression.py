"""Deterministic Compress(d, q; h, L) used by paper equation (6)."""
import re


def terms(text):
    return {word.lower() for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text or '') if len(word) > 2}


def compress_each(context, query, max_sentences, max_chars):
    if max_sentences < 1 or max_chars < 1:
        raise ValueError('Compression limits must be positive')
    query_terms = terms(query)
    compressed = []
    for document in context:
        # Preserve newline boundaries while normalizing whitespace inside sentences.
        sentences = [' '.join(s.split()) for s in re.split(r'(?<=[.!?])\s+|(?<=[。！？])|[\r\n]+', str(document or '')) if s.strip()]
        ranked = sorted(range(len(sentences)), key=lambda i: (-len(terms(sentences[i]) & query_terms), i))
        selected_indices = sorted(ranked[:max_sentences])
        text = ' '.join(sentences[i] for i in selected_indices)[:max_chars]
        if text:
            compressed.append(text)
    return compressed
