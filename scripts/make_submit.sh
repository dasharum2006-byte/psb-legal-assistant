#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../submit"
rm -f ../submission.zip
zip -r ../submission.zip .
echo "Готово: submission.zip (solution.py и requirements.txt в корне)"
