"""Sentence/location metadata and offline support mapping into an existing corpus."""
from collections import defaultdict
import re
from reasonroute.evidence import EvidenceCatalog


def normalized(text):
    return ' '.join(str(text or '').casefold().split())


def normalized_positions(text):
    output, positions, pending_space = [], [], None
    for index, char in enumerate(text):
        if char.isspace():
            if output:
                pending_space = index
            continue
        if pending_space is not None:
            output.append(' ')
            positions.append(pending_space)
            pending_space = None
        for lower in char.casefold():
            output.append(lower)
            positions.append(index)
    return ''.join(output), positions


def sentence_spans(text):
    start = 0
    for match in re.finditer(r'(?<=[.!?])\s+|(?<=[。！？])|[\r\n]+', text):
        end = match.start()
        if text[start:end].strip():
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            yield left, end - (len(text[start:end]) - len(text[start:end].rstrip()))
        start = match.end()
    if text[start:].strip():
        left = start + len(text[start:]) - len(text[start:].lstrip())
        yield left, len(text.rstrip())


def paragraph_records(document):
    paragraphs = document.get('paragraphs')
    if paragraphs is None:
        paragraphs = re.split(r'\n\s*\n', str(document.get('text') or ''))
    for index, raw in enumerate(paragraphs):
        item = raw if isinstance(raw, dict) else {'text': str(raw)}
        if 'sentences' in item:
            sentences = item['sentences']
            text, spans, cursor = '', [], 0
            for sid, sentence in enumerate(sentences):
                content = str(sentence.get('text') or '') if isinstance(sentence, dict) else str(sentence)
                if text:
                    text += ' '
                cursor = len(text)
                text += content
                if content.strip():
                    spans.append((int(sentence.get('sentence_id', sid)) if isinstance(sentence, dict) else sid,
                                  cursor, len(text)))
        else:
            text = str(item.get('text') or '')
            spans = [(sid, start, end) for sid, (start, end) in enumerate(sentence_spans(text))]
        if text.strip():
            yield int(item.get('paragraph_id', index)), text, spans


def document_chunks(document, size, overlap):
    doc_id = str(document['doc_id'])
    seen_paragraphs = set()
    for paragraph_id, text, spans in paragraph_records(document):
        if paragraph_id in seen_paragraphs:
            raise ValueError('Duplicate paragraph ID within a document')
        seen_paragraphs.add(paragraph_id)
        for index, start in enumerate(range(0, len(text), size - overlap)):
            end = min(start + size, len(text))
            selected = [(sid, left, right) for sid, left, right in spans if left < end and right > start]
            if not selected:
                continue
            ids = [sid for sid, _, _ in selected]
            yield {'doc_id': doc_id, 'chunk_id': f'{doc_id}__p{paragraph_id:04d}__c{index:05d}',
                   'title': document.get('title', ''), 'content': text[start:end],
                   'links': document.get('links', []), 'corpus': 'full_document_snapshot',
                   'dataset_refs': document.get('dataset_refs', []),
                   'metadata': {'paragraph_id': paragraph_id, 'chunk_index': index,
                                'sentence_start': min(ids), 'sentence_end': max(ids) + 1,
                                'sentence_ids': ids, 'sentence_spans': [list(x) for x in selected],
                                'char_start': start, 'char_end': end, 'char_count': end - start,
                                'source_dataset': document.get('source_dataset'),
                                'source_url': document.get('source_url', '')}}
            if end == len(text):
                break


class SupportMapper:
    """Map annotations offline; never add, rewrite or expose support flags in the KB."""

    def __init__(self, chunks):
        self.by_title, self.by_id = defaultdict(list), defaultdict(list)
        for chunk in chunks:
            self.by_title[EvidenceCatalog.key(chunk.get('title') or '')].append(chunk)
            self.by_id[str(chunk['doc_id'])].append(chunk)

    @staticmethod
    def paragraphs(chunks):
        grouped = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.get('metadata', {}).get('paragraph_id', 0)].append(chunk)
        output = []
        for pid, group in grouped.items():
            group.sort(key=lambda c: c['metadata']['char_start'])
            text, complete = '', True
            for chunk in group:
                start = chunk['metadata']['char_start']
                if start > len(text):
                    complete = False
                    break
                old_end = min(len(text), start + len(chunk['content']))
                if text[start:old_end] != chunk['content'][:old_end - start]:
                    raise ValueError('Inconsistent overlapping corpus chunks')
                text += chunk['content'][max(0, len(text) - start):]
            if complete:
                output.append((pid, text, group))
        return output

    def map_one(self, reference):
        candidates = self.by_id.get(str(reference.get('doc_id')), []) if reference.get('doc_id') else self.by_title.get(EvidenceCatalog.key(reference.get('title') or ''), [])
        docs = defaultdict(list)
        for chunk in candidates:
            docs[chunk['doc_id']].append(chunk)
        output = []
        for doc_id, chunks in docs.items():
            for pid, text, group in self.paragraphs(chunks):
                if reference.get('paragraph_id') is not None and int(reference['paragraph_id']) != pid:
                    continue
                needle = reference.get('text')
                alignment = reference.get('alignment', 'exact_support_text')
                if needle:
                    flat, offsets = normalized_positions(text)
                    query = normalized(needle)
                    if alignment == 'approximate_answer_position' and query:
                        match = re.search(r'(?<!\w)' + re.escape(query) + r'(?!\w)', flat)
                        location = match.start() if match else -1
                    else:
                        location = flat.find(query) if query else -1
                    if location < 0:
                        continue
                    left, right = offsets[location], offsets[location + len(query) - 1] + 1
                    matched = [c for c in group if c['metadata']['char_start'] < right and c['metadata']['char_end'] > left]
                    matched_sentence_ids = {sid for c in matched for sid, a, b in c['metadata']['sentence_spans']
                                            if a < right and b > left}
                elif reference.get('sentence_ids') is not None:
                    requested = set(reference['sentence_ids'])
                    matched = [c for c in group if requested & set(c['metadata']['sentence_ids'])]
                    if not requested <= {sid for c in matched for sid in c['metadata']['sentence_ids']}:
                        continue
                    alignment = 'source_sentence_id'
                    matched_sentence_ids = requested
                else:
                    continue
                output.append({'doc_id': doc_id, 'title': group[0].get('title', ''),
                               'paragraph_id': pid, 'chunk_ids': [c['chunk_id'] for c in matched],
                               'sentence_ids': sorted(matched_sentence_ids),
                               'annotation_sentence_ids': reference.get('sentence_ids', []),
                               'alignment': alignment})
        return output

    def map_all(self, references):
        evidence, missing = [], []
        for ref in references:
            matched = self.map_one(ref)
            if not matched:
                missing.append({'title': ref.get('title'), 'alignment': ref.get('alignment')})
            evidence.extend(matched)
        unique = {}
        for item in evidence:
            key = (item['doc_id'], item['paragraph_id'], tuple(item['chunk_ids']))
            unique.setdefault(key, item)
        return list(unique.values()), missing


def answer_values(value):
    if isinstance(value, dict):
        spans = [s.get('text', '') for s in value.get('answer_spans') or []]
        return answer_values(spans or value.get('answer_value') or value.get('answer'))
    if isinstance(value, list):
        return list(dict.fromkeys(a for x in value for a in answer_values(x)))
    return [str(value).strip()] if value is not None and str(value).strip() else []


def adapt_question(row, dataset, article=None):
    question = row.get('question') or row.get('question_text')
    question = question.get('text', '') if isinstance(question, dict) else question
    refs = []
    answers = answer_values(row.get('answer') or row.get('answers'))
    if dataset in ('hotpotqa', '2wikimultihopqa'):
        context = row.get('context') or []
        contexts = dict(zip(context.get('title', []), context.get('sentences', []))) if isinstance(context, dict) else dict(context)
        support = row.get('supporting_facts') or []
        support = zip(support.get('title', []), support.get('sent_id', [])) if isinstance(support, dict) else support
        for title, sid in support:
            sentences = contexts.get(title, [])
            if not 0 <= int(sid) < len(sentences):
                raise ValueError('Supporting sentence index is not present in the source context')
            refs.append({'title': title, 'text': sentences[int(sid)], 'sentence_ids': [int(sid)],
                         'alignment': 'exact_support_text'})
        if not refs and dataset == '2wikimultihopqa':
            # Relation triples alone do not identify a supporting text span.
            raise ValueError('2Wiki support sentences are required; evidence triples cannot invent chunk mappings')
    elif dataset == 'musique':
        if row.get('answerable') is False:
            raise ValueError('Unanswerable MuSiQue question')
        for paragraph in row.get('paragraphs') or []:
            if paragraph.get('is_supporting'):
                refs.append({'title': paragraph['title'], 'text': paragraph['paragraph_text'],
                             'alignment': 'exact_support_text'})
        answers = answer_values([row.get('answer'), *(row.get('answer_aliases') or [])])
    elif dataset == 'iirc':
        if article is None:
            raise ValueError('IIRC question must retain its parent article metadata')
        for context in row.get('context') or []:
            title = article.get('title') if context.get('passage') == 'main' else context.get('passage')
            refs.append({'title': title, 'text': context.get('text'), 'alignment': 'exact_support_text'})
    elif dataset == 'naturalquestions':
        annotations = row.get('annotations') or {}
        document = row.get('document') or {}
        tokens = document.get('tokens') or row.get('document_tokens') or []
        words = tokens.get('token', []) if isinstance(tokens, dict) else [t.get('token', '') for t in tokens]
        html = tokens.get('is_html', []) if isinstance(tokens, dict) else [t.get('html_token', False) for t in tokens]
        def token_span(span):
            start, end = int(span.get('start_token', -1)), int(span.get('end_token', -1))
            if not 0 <= start < end <= len(words):
                return ''
            return ' '.join(words[i] for i in range(start, end) if not (html[i] if i < len(html) else False))
        if isinstance(annotations, dict):
            answers = answer_values([group.get('text', []) for group in annotations.get('short_answers') or []])
            if not answers:
                answers = ['yes' if value == 1 else 'no' for value in annotations.get('yes_no_answer') or [] if value in (0, 1)]
            long_support = [token_span(span) for span in annotations.get('long_answer') or []]
        else:
            answers = answer_values([token_span(span) for group in annotations for span in group.get('short_answers') or []])
            if not answers:
                answers = [group['yes_no_answer'].lower() for group in annotations if group.get('yes_no_answer') in ('YES', 'NO')]
            long_support = [token_span(group.get('long_answer') or {}) for group in annotations]
        title = document.get('title') or row.get('document_title')
        # With no sentence-level standard support, locate reference answers in the existing article.
        refs = [{'title': title, 'text': support, 'alignment': 'approximate_context_position'} for support in long_support if support]
        if not refs and not any(a in ('yes', 'no') for a in answers):
            refs = [{'title': title, 'text': answer, 'alignment': 'approximate_answer_position'} for answer in answers]
    else:
        raise ValueError('Unsupported dataset adapter')
    if not question or not answers or not refs:
        raise ValueError('Question, reference answer or supporting annotations are missing')
    return {'id': str(row.get('id') or row.get('_id') or row.get('qid') or ''),
            'question': question, 'answers': answers, 'dataset': dataset, 'references': refs,
            'reference_policy': 'any' if dataset == 'naturalquestions' else 'all'}
