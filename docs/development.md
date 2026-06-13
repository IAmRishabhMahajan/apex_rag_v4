# Development Commands

Use `uv` for Python environment management. On this Codex desktop workspace, the bundled Python runtime is available at:

```powershell
C:\Users\risha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

Recommended local commands:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
$env:UV_PYTHON_INSTALL_DIR='.uv-python'
uv run --python 'C:\Users\risha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' python -m unittest discover -s tests
ruff format --check .
ruff check .
mypy tests
```

The workspace-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` keep generated environment files inside the project boundary.

