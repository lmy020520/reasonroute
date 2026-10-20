"""Title, paragraph-neighbor and explicit document-link context expansion."""
from collections import defaultdict
import json
import re
from urllib.parse import unquote


class EvidenceCatalog:
    def __init__(self, records, include_titles=False):
        self.rows = []
        self.text_indices = defaultdict(list)
        self.documents = defaultdict(list)
        self.targets = defaultdict(list)
        for record in records:
            text = record.get('content') or record.get('text') or record.get('body') or ''
            if not text:
                continue
            metadata = record.get('metadata') or {}
            title = record.get('title') or metadata.get('title') or ''
            rendered = f'Title: {title}\n{text}' if include_titles and title and title.lower() not in text[:160].lower() else text
            doc_id = str(record.get('doc_id') or metadata.get('doc_id') or title or record.get('chunk_id') or len(self.rows))
            index = len(self.rows)
            self.rows.append({'record': record, 'text': rendered, 'document': doc_id,
                              'position': tuple(int(metadata.get(k) or 0) for k in ('paragraph_id', 'chunk_index', 'sentence_start'))})
            self.text_indices[rendered].append(index)
            self.documents[doc_id].append(index)
            for target in {doc_id, title, record.get('chunk_id', '')}:
                if target:
                    self.targets[self.key(target)].append(index)
        for indices in self.documents.values():
            indices.sort(key=lambda i: (self.rows[i]['position'], i))
        self.locations = {idx: position for indices in self.documents.values() for position, idx in enumerate(indices)}

    @staticmethod
    def key(value):
        value = str(value)
        if '/wiki/' in value:
            value = value.split('/wiki/', 1)[1]
        return ' '.join(unquote(value).replace('_', ' ').lower().split())

    def links(self, record):
        metadata = record.get('metadata') or {}
        targets = []
        for source in (record, metadata):
            for field in ('links', 'linked_doc_ids', 'linked_titles'):
                values = source.get(field) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    if isinstance(value, dict):
                        value = value.get('doc_id') or value.get('title') or value.get('target') or value.get('url')
                    if value:
                        targets.append(self.key(value))
        text = record.get('content') or record.get('text') or record.get('body') or ''
        targets.extend(self.key(value.split('|', 1)[0]) for value in re.findall(r'\[\[([^\]]+)\]\]', text))
        targets.extend(self.key(value) for value in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text))
        return targets

    def expand(self, seed_context, limit=20):
        indices = []
        for text in seed_context:
            for idx in self.text_indices.get(text, []):
                row = self.rows[idx]
                document = self.documents[row['document']]
                position = self.locations[idx]
                indices.extend(document[max(0, position - 1):position + 2])
                # The first paragraph supplies title-related document background.
                indices.extend(document[:1])
                for target in self.links(row['record']):
                    indices.extend(self.targets.get(target, [])[:3])
        output, seen = [], set(seed_context)
        for index in indices:
            text = self.rows[index]['text']
            if text not in seen:
                seen.add(text)
                output.append(text)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def from_jsonl(cls, path, include_titles=False):
        with open(path, encoding='utf-8') as handle:
            return cls((json.loads(line) for line in handle if line.strip()), include_titles)
