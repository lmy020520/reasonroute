"""Offline import-order and CLI checks for the public API-backed method."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))


CLI = ('scripts.run_reasonroute', 'tools.train_method_models', 'tools.build_method_index',
       'tools.prepare_method_data', 'tools.build_fixed_corpus', 'tools.augment_explanation_from_iirc',
       'tools.reclassify_public_eval', 'tools.build_classifier_training_dataset',
       'tools.build_compact_gate_training_dataset')

IMPORT_GUARD = '''
import importlib, importlib.abc, json, socket, sys
import requests
class NoLocalModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers', 'faiss'}:
            raise ModuleNotFoundError('Optional local-model import attempted: ' + fullname)
def no_network(*args, **kwargs):
    raise RuntimeError('Network access attempted during import')
sys.meta_path.insert(0, NoLocalModels())
socket.create_connection = no_network
socket.socket.connect = no_network
requests.sessions.Session.request = no_network
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
print('imports passed')
'''


def check():
    modules = ['evaluate_rag']
    for package in ('reasonroute', 'tools', 'scripts'):
        modules.append(package)
        modules.extend(f'{package}.{path.stem}' for path in sorted((ROOT / package).glob('*.py'))
                       if path.stem != '__init__')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               DEEPSEEK_API_KEY='', SILICONFLOW_API_KEY='')
    cases = {}

    def run(name, command, cwd):
        process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
        cases[name] = {'passed': process.returncode == 0}
        if process.returncode:
            cases[name]['stderr'] = process.stderr[-3000:]
            cases[name]['stdout'] = process.stdout[-1000:]

    orders = [modules,
              ['tools.rag_resarch_tool', 'evaluate_rag', 'reasonroute.retrieval', 'reasonroute.branch_flow'],
              ['reasonroute.branch_flow', 'reasonroute.retrieval', 'evaluate_rag', 'tools.rag_resarch_tool'],
              ['tools.reclassify_public_eval', 'reasonroute.annotation', 'tools.build_classifier_training_dataset']]
    for index, order in enumerate(orders):
        run(f'import_order_{index}', [sys.executable, '-c', IMPORT_GUARD, json.dumps(order)], ROOT)
    with tempfile.TemporaryDirectory() as outside:
        for module in CLI:
            run(module + ':module', [sys.executable, '-m', module, '--help'], ROOT)
            run(module + ':script', [sys.executable, str(ROOT / (module.replace('.', '/') + '.py')), '--help'], outside)
    return {'passed': all(case['passed'] for case in cases.values()), 'modules_checked': len(modules),
            'network_during_import': 'forbidden', 'optional_local_model_imports': 'forbidden',
            'cases': cases, 'scope': 'Offline imports and CLI parsing, not live API execution'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
