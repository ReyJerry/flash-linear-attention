# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run mandatory CPU/GPU suites and write a benchmark acceptance manifest."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def code_digest():
    paths = sorted((ROOT / 'fla/ops/phba').rglob('*.py'))
    paths += sorted((ROOT / 'fla/models/phba').rglob('*.py'))
    paths += [ROOT / 'fla/layers/phba.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get('passed') or manifest.get('code_digest') != code_digest():
        raise RuntimeError('missing/stale GPU acceptance; rerun run_decode_acceptance.py')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    result = {'passed': False, 'code_digest': code_digest(), 'commands': []}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; GPU acceptance cannot pass')
        result['torch'] = torch.__version__
        result['gpu'] = torch.cuda.get_device_name()
        result['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        suites = [
            [sys.executable, 'tests/ops/test_phba_decode_cpu.py'],
            [sys.executable, 'tests/ops/test_phba_decode_kernels_cpu.py'],
            [sys.executable, '-m', 'pytest', '-q', 'tests/ops/test_phba_decode.py',
             'tests/models/test_phba_cached_inference.py', f'--junitxml={directory / "decode.xml"}'],
            [sys.executable, '-m', 'pytest', '-q', 'tests/ops/test_phba_ablation.py',
             'tests/models/test_modeling_phba_ablation.py', 'tests/ops/test_phba.py',
             'tests/models/test_modeling_phba.py', 'tests/layers/test_phba_linear.py',
             f'--junitxml={directory / "regression.xml"}'],
        ]
        for number, command in enumerate(suites):
            with (directory / f'suite_{number}.log').open('w') as log:
                run = subprocess.run(command, cwd=ROOT, env={**os.environ, 'FLA_CI_ENV': '0'},
                                     stdout=log, stderr=subprocess.STDOUT)
            result['commands'].append({'argv': command, 'returncode': run.returncode})
            if run.returncode:
                raise RuntimeError(f'suite {number} failed; see suite_{number}.log')
        cases = list(ET.parse(directory / 'decode.xml').getroot().iter('testcase'))
        if not cases or any(c.find('skipped') is not None for c in cases):
            raise RuntimeError('decode GPU cases were skipped or not collected')
        result['gpu_decode_cases'] = len(cases)
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    (directory / 'acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
