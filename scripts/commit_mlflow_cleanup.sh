#!/usr/bin/env bash
# Execute MLflow cleanup - Final commit script
# This script performs the actual git operations to complete the cleanup

set -e

echo "🚀 MLflow Cleanup - Final Commit"
echo "================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "modules" ]; then
    echo -e "${RED}❌ Error: Must be run from repository root${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Verify current git status${NC}"
echo "-----------------------------------"
git status --short
echo ""

# Check if mlflow.db is tracked
echo -e "${BLUE}Step 2: Check if mlflow.db is tracked${NC}"
echo "---------------------------------------"
if git ls-files | grep -q "^mlflow.db$"; then
    echo -e "${YELLOW}⚠️  mlflow.db is currently tracked in git${NC}"
    echo ""
    
    read -p "Remove mlflow.db from git tracking? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing mlflow.db from git tracking...${NC}"
        git rm --cached mlflow.db
        echo -e "${GREEN}✅ mlflow.db removed from tracking${NC}"
    else
        echo -e "${RED}❌ Aborted. Please remove mlflow.db manually: git rm --cached mlflow.db${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ mlflow.db is not tracked${NC}"
fi
echo ""

# Make cleanup script executable
echo -e "${BLUE}Step 3: Make cleanup script executable${NC}"
echo "---------------------------------------"
if [ -f "scripts/clean_artifacts.sh" ]; then
    chmod +x scripts/clean_artifacts.sh
    chmod +x scripts/verify_mlflow_cleanup.sh
    echo -e "${GREEN}✅ Scripts are now executable${NC}"
else
    echo -e "${RED}❌ scripts/clean_artifacts.sh not found${NC}"
    exit 1
fi
echo ""

# Stage all changes
echo -e "${BLUE}Step 4: Stage all changes${NC}"
echo "-------------------------"
git add .gitignore
git add Makefile
git add README.md
git add docs/getting-started/setup.md
git add docs/getting-started/mlflow-tracking.md
git add mkdocs.yml
git add scripts/clean_artifacts.sh
git add scripts/verify_mlflow_cleanup.sh
git add MLFLOW_CLEANUP_SUMMARY.md
echo -e "${GREEN}✅ Changes staged${NC}"
echo ""

# Show what will be committed
echo -e "${BLUE}Step 5: Review changes${NC}"
echo "----------------------"
git status --short
echo ""

# Ask for confirmation
read -p "Proceed with commit? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ Commit aborted${NC}"
    exit 1
fi

# Commit changes
echo -e "${BLUE}Step 6: Commit changes${NC}"
echo "----------------------"
git commit -m "chore: remove mlflow.db from version control and enhance artifact management

- Remove mlflow.db from git tracking (gitignored, generated at runtime)
- Consolidate .gitignore MLflow entries (remove duplicates)
- Add comprehensive MLflow artifact patterns (*.db-shm, *.db-wal, .mlflow/)
- Enhance 'make clean' to remove MLflow artifacts
- Add MLflow tracking documentation (docs/getting-started/mlflow-tracking.md)
- Update setup docs to clarify MLflow is optional
- Create cleanup script (scripts/clean_artifacts.sh) with dry-run mode
- Add verification script (scripts/verify_mlflow_cleanup.sh)
- Update mkdocs navigation to include MLflow guide
- Add implementation summary (MLFLOW_CLEANUP_SUMMARY.md)

MLflow is now fully optional. Local DB and artifacts are generated at runtime
and never committed. Use 'make clean' or './scripts/clean_artifacts.sh' to
remove local tracking artifacts.

Fixes: MLflow artifacts cleanup
See: MLFLOW_CLEANUP_SUMMARY.md for details"

echo -e "${GREEN}✅ Changes committed${NC}"
echo ""

# Show commit
echo -e "${BLUE}Step 7: Verify commit${NC}"
echo "--------------------"
git log --oneline -1
git show --stat HEAD
echo ""

# Final steps
echo -e "${GREEN}✅ Commit complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Run tests: make test"
echo "  2. Run linting: make lint"
echo "  3. Verify docs build: mkdocs build"
echo "  4. Push changes: git push origin $(git branch --show-current)"
echo ""
echo -e "${BLUE}Optional verification:${NC}"
echo "  ./scripts/verify_mlflow_cleanup.sh"
echo ""
