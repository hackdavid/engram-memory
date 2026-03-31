@echo off
REM Launcher for Windows when engram-e2e.exe is not on PATH (common with conda/base Python).
REM Usage: run from repo root, or pass full path. Requires: pip install -e .
python -m engram.cli.e2e_validate %*
