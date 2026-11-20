#!/bin/bash

set -u

REPORT="cardiopredict_diagnostic.txt"

{
echo "======================================"
echo " CARDIOPREDICT-X PROJECT DIAGNOSTIC"
echo "======================================"

echo
echo "1. PROJECT LOCATION"
pwd

echo
echo "2. PROJECT FILES"
find . -maxdepth 4 -type f \
  -not -path "*/.venv/*" \
  -not -path "*/venv/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/.git/*" \
  -not -name ".env" \
  -not -name "*.csv" \
  -not -name "*.pkl" \
  -not -name "*.joblib" \
  -not -name "*.json" \
  | sort | head -150

echo
echo "3. PYTHON ENVIRONMENT"
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python --version 2>&1
python -m pip --version 2>&1

echo
echo "4. REQUIRED PACKAGES"
for package in fastapi uvicorn scikit-learn \
  pandas numpy xgboost joblib; do
  python -m pip show "$package" 2>/dev/null \
    | grep -E "^(Name|Version):" \
    || echo "$package: not installed"
done

echo
echo "5. MODEL FILES"
find . -type f \( \
  -name "*.pkl" -o \
  -name "*.joblib" -o \
  -name "*.pt" \
\) -not -path "*/.venv/*" \
  -not -path "*/venv/*"

echo
echo "6. BACKEND ENTRY POINTS"
find . -type f \( \
  -name "main.py" -o \
  -name "app.py" -o \
  -name "server.py" \
\) -not -path "*/.venv/*" \
  -not -path "*/venv/*" \
  -not -path "*/node_modules/*"

echo
echo "7. FRONTEND"
if [ -f "frontend/package.json" ]; then
  echo "Frontend package.json found"
elif [ -f "package.json" ]; then
  echo "Root package.json found"
else
  echo "No Node.js package.json found"
fi

echo
echo "8. TEST FILES"
find . -type f \( \
  -name "test_*.py" -o \
  -name "*_test.py" \
\) -not -path "*/.venv/*" \
  -not -path "*/venv/*"

echo
echo "9. GIT STATUS"
if [ -d ".git" ]; then
  git status --short
else
  echo "Git repository not initialized"
fi

echo
echo "DIAGNOSTIC COMPLETED"

} > "$REPORT" 2>&1

echo "Report saved to: $REPORT"
