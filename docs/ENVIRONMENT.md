# Environment and resource notes

The paired source is FuxiCTR 2.3.10 with the checked-in numerical-feature
integration. The recorded remote interpreter was Python 3.10 with PyTorch
2.6.0, NumPy 1.24.0, pandas 2.0.1, PyArrow 24.0.0, scikit-learn 1.7.2,
PyYAML 6.0.2, and packaging 24.2. This is an observed configuration, not a
tested compatibility matrix. Install a PyTorch wheel appropriate for the
host's CUDA driver before `pip install -r requirements.txt` if necessary.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python reproduction/tools/check_environment.py --require-cuda
python reproduction/tools/smoke_imports.py
python -m unittest discover -s reproduction/tests -v
```

The RTDL 0.0.12 implementation is vendored under `method/vendor/`, so a second
RTDL installation is not required. The benchmark imports its bundled copy
explicitly even if a global installation exists.
The same principle applies to FuxiCTR: invoke `backbone/run_expid.py` from this
repository root, where the bundled `fuxictr/` package is first on `sys.path`.

The full 1,458-task matrix is resource-intensive. Before launching it, inspect
available GPUs, memory, storage capacity, and the data-loader worker count in
the YAML. Start with one method/backbone/seed and check its log and output.
The launcher writes checkpoint files under `outputs/checkpoints/` by default;
those files are excluded by `.gitignore` and should not be committed. The
large source datasets likewise belong outside the repository. A process per
GPU avoids implicit shared-memory/data-parallel assumptions.

`reproduction/tools/check_environment.py --allow-missing-packages` verifies the code
layout on an analysis-only machine. It does **not** certify that a training
environment is ready. For training, omit that flag and optionally add
`--require-cuda`.
