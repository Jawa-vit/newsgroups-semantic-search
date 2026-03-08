#!/usr/bin/env bash
# setup.sh — One-command environment setup
# Usage: bash setup.sh

set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Newsgroups Semantic Search — Environment Setup            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
PY=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY" | cut -d. -f1)
MINOR=$(echo "$PY" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
  echo "ERROR: Python 3.9+ required (found $PY)"
  exit 1
fi
echo "✓ Python $PY"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment ..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "✓ Virtual environment activated"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "→ Installing dependencies ..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p db/chroma models analysis
echo "✓ Directories created"

# ── Dataset check ─────────────────────────────────────────────────────────────
if [ ! -d "data/20_newsgroups" ]; then
  echo ""
  echo "⚠  Dataset not found at data/20_newsgroups/"
  echo "   Please unzip your dataset there:"
  echo "   mkdir -p data && tar -xzf 20_newsgroups.tar.gz -C data/"
  echo ""
else
  CATS=$(ls data/20_newsgroups/ | wc -l)
  echo "✓ Dataset found ($CATS categories)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║   1. Index the corpus (run once, ~5 min on CPU):            ║"
echo "║      python scripts/ingest.py                               ║"
echo "║                                                              ║"
echo "║   2. Train clusters (run once, ~10 min on CPU):             ║"
echo "║      python scripts/cluster.py                              ║"
echo "║                                                              ║"
echo "║   3. Generate cluster report (optional):                    ║"
echo "║      python analysis/cluster_report.py                      ║"
echo "║                                                              ║"
echo "║   4. Start the API server:                                  ║"
echo "║      uvicorn app.main:app --host 0.0.0.0 --port 8000       ║"
echo "║                                                              ║"
echo "║   5. Open API docs:                                         ║"
echo "║      http://localhost:8000/docs                             ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
