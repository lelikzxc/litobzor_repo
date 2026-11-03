"""Execute the EDA with a real Jupyter kernel and retain its visible outputs."""
from pathlib import Path
import hashlib
import json
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'gipoteza/EDA_IC_SEM_M2.ipynb'

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
before = {
    str(p.relative_to(ROOT)): sha(p)
    for p in [ROOT / 'pipeline.ipynb',
              ROOT / 'gipoteza/sem_architecture_pilot_20261102/DATA.npz',
              ROOT / 'gipoteza/sem_architecture_pilot_20261102/DATA_MANIFEST.json']
}
start = time.monotonic()
notebook = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='ds-3.11',
                        resources={'metadata': {'path': str(ROOT / 'gipoteza')}},
                        allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(notebook, NOTEBOOK)

nbformat.validate(notebook)
code = [c for c in notebook.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code)
assert not [o for c in code for o in c.outputs if o.output_type == 'error']
assert all(sha(ROOT / name) == value for name, value in before.items())
png_count = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
result = {
    'status': 'PASS_EXECUTED_NOTEBOOK',
    'notebook': str(NOTEBOOK.relative_to(ROOT)),
    'cells': len(notebook.cells), 'executed_code_cells': len(code),
    'embedded_figures': png_count,
    'seconds': time.monotonic() - start,
    'kernel': 'ds-3.11',
    'notebook_sha256': sha(NOTEBOOK),
    'unchanged_input_sha256': before,
}
(FOLDER / 'EXECUTION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
