#!/usr/bin/env bash
# Quick verification script for MLflow cleanup changes
# Run this before committing to ensure everything is correct

set -e

echo "🔍 MLflow Cleanup Verification"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

# Check 1: mlflow.db is tracked in git (should be removed)
echo -n "1. Checking if mlflow.db is tracked in git... "
if git ls-files | grep -q "^mlflow.db$"; then
    echo -e "${RED}FAIL${NC}"
    echo "   ❌ mlflow.db is still tracked. Run: git rm --cached mlflow.db"
    FAIL=$((FAIL + 1))
else
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ mlflow.db is not tracked"
    PASS=$((PASS + 1))
fi
echo ""

# Check 2: .gitignore contains MLflow patterns
echo -n "2. Checking .gitignore for MLflow patterns... "
if grep -q "mlflow.db" .gitignore && grep -q "mlruns/" .gitignore; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ .gitignore contains MLflow patterns"
    PASS=$((PASS + 1))
else
    echo -e "${RED}FAIL${NC}"
    echo "   ❌ .gitignore missing MLflow patterns"
    FAIL=$((FAIL + 1))
fi
echo ""

# Check 3: Makefile clean target includes MLflow
echo -n "3. Checking Makefile clean target... "
if grep -A 3 "^clean:" Makefile | grep -q "mlflow"; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ Makefile clean includes MLflow artifacts"
    PASS=$((PASS + 1))
else
    echo -e "${RED}FAIL${NC}"
    echo "   ❌ Makefile clean missing MLflow cleanup"
    FAIL=$((FAIL + 1))
fi
echo ""

# Check 4: MLflow documentation exists
echo -n "4. Checking MLflow documentation... "
if [ -f "docs/getting-started/mlflow-tracking.md" ]; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ MLflow documentation exists"
    PASS=$((PASS + 1))
else
    echo -e "${RED}FAIL${NC}"
    echo "   ❌ MLflow documentation missing"
    FAIL=$((FAIL + 1))
fi
echo ""

# Check 5: mkdocs.yml includes MLflow doc
echo -n "5. Checking mkdocs.yml navigation... "
if grep -q "mlflow-tracking.md" mkdocs.yml; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ mkdocs.yml includes MLflow guide"
    PASS=$((PASS + 1))
else
    echo -e "${RED}FAIL${NC}"
    echo "   ❌ mkdocs.yml missing MLflow guide"
    FAIL=$((FAIL + 1))
fi
echo ""

# Check 6: Cleanup script exists and is executable
echo -n "6. Checking cleanup script... "
if [ -f "scripts/clean_artifacts.sh" ]; then
    if [ -x "scripts/clean_artifacts.sh" ]; then
        echo -e "${GREEN}PASS${NC}"
        echo "   ✅ Cleanup script exists and is executable"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}PARTIAL${NC}"
        echo "   ⚠️  Script exists but not executable. Run: chmod +x scripts/clean_artifacts.sh"
        PASS=$((PASS + 1))
    fi
else
    echo -e "${RED}FAIL${NC}"
    echo "   ❌ Cleanup script missing"
    FAIL=$((FAIL + 1))
fi
echo ""

# Check 7: No duplicate MLflow entries in .gitignore
echo -n "7. Checking for duplicate MLflow entries in .gitignore... "
MLFLOW_COUNT=$(grep -c "^# MLflow" .gitignore || echo "0")
if [ "$MLFLOW_COUNT" -eq "0" ]; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ No duplicate MLflow sections (consolidated)"
    PASS=$((PASS + 1))
elif [ "$MLFLOW_COUNT" -eq "1" ]; then
    echo -e "${GREEN}PASS${NC}"
    echo "   ✅ Single MLflow section (consolidated)"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "   ⚠️  Multiple MLflow sections found ($MLFLOW_COUNT)"
    PASS=$((PASS + 1))
fi
echo ""

# Summary
echo "=============================="
echo "Summary: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: git rm --cached mlflow.db  (if needed)"
    echo "  2. Run: chmod +x scripts/clean_artifacts.sh"
    echo "  3. Run: make test"
    echo "  4. Run: make lint"
    echo "  5. Commit changes"
    exit 0
else
    echo -e "${RED}❌ Some checks failed. Please fix the issues above.${NC}"
    exit 1
fi
