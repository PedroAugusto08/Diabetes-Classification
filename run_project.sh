#!/usr/bin/env bash
set -euo pipefail

# Diretório onde este script está localizado.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/Diabetes-Classification"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Erro: python3 não encontrado no PATH."
  exit 1
fi

echo "Instalando dependências..."
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Executando projeto..."
python3 "$PROJECT_DIR/src/main.py" "$@"
