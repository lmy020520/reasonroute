"""Offline pre-publication checks; prints locations but never matched values."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{8,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[A-Z0-9]{16}",
    r"AIza[A-Za-z0-9_-]{25,}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
)
PRIVATE_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]lmyproject[\\/]|[\\/]Users[\\/]|/\x68ome/lmy/|/mnt/[a-z]/lmyproject/")


def audit() -> dict:
    findings = []
    py_count = json_count = byte_count = 0
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        files.append(rel)
        raw = path.read_bytes()
        byte_count += len(raw)
        if '.git' in path.parts or path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
            findings.append({'path': rel, 'kind': 'private_metadata_or_environment_file'})
        if len(raw) > 5_000_000:
            findings.append({'path': rel, 'kind': 'unexpected_large_file'})
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            findings.append({'path': rel, 'kind': 'unexpected_binary'})
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(re.search(pattern, line) for pattern in SECRET_PATTERNS):
                findings.append({'path': rel, 'line': number, 'kind': 'credential_pattern'})
            if PRIVATE_PATH.search(line):
                findings.append({'path': rel, 'line': number, 'kind': 'personal_absolute_path'})
        if path.suffix == '.py':
            py_count += 1
            try:
                tree = ast.parse(text, filename=rel)
                compile(tree, rel, 'exec')
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    names = [n.id for n in node.targets if isinstance(n, ast.Name)]
                    if any(re.search(r'(API_KEY|ACCESS_TOKEN|PASSWORD|CLIENT_SECRET)$', name, re.I) for name in names):
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and node.value.value:
                            findings.append({'path': rel, 'line': node.lineno, 'kind': 'hardcoded_credential_assignment'})
            except SyntaxError:
                findings.append({'path': rel, 'kind': 'python_syntax_error'})
        elif path.suffix == '.json':
            json_count += 1
            try:
                json.loads(text)
            except ValueError:
                findings.append({'path': rel, 'kind': 'invalid_json'})
    return {'files': len(files), 'python_files': py_count, 'json_files': json_count,
            'bytes': byte_count, 'passed': not findings, 'findings': findings,
            'scope': 'selected review bundle only; no git history or excluded files'}


if __name__ == '__main__':
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
