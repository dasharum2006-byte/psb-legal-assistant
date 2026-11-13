#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../submit"
rm -f ../submission.zip
rm -rf __pycache__
zip -r ../submission.zip solution.py requirements.txt
echo "Готово: submission.zip (solution.py и requirements.txt в корне)"
