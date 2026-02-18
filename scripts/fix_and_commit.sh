#!/usr/bin/env bash
# Fix and complete MLflow cleanup
# Run this script to properly execute all steps

set -e

echo "🔧 MLflow Cleanup - Step by Step Fix"
echo "====================================="
echo ""

# Step 1: Remove mlflow.db from git tracking
echo "Step 1: Removing mlflow.db from git tracking..."
if git ls-files | grep -q "^mlflow.db$"; then
    git rm --cached mlflow.db
    echo "✅ mlflow.db removed from git index"
else
    echo "✅ mlflow.db already not tracked"
fi
echo ""

# Step 2: Make scripts executable
echo "Step 2: Making scripts executable..."
chmod +x scripts/clean_artifacts.sh
chmod +x scripts/verify_mlflow_cleanup.sh
chmod +x scripts/commit_mlflow_cleanup.sh
echo "✅ Scripts are executable"
echo ""

# Step 3: Stage changes
echo "Step 3: Staging all changes..."
git add .gitignore
git add Makefile
git add README.md
git add docs/getting-started/setup.md
git add docs/getting-started/mlflow-tracking.md
git add mkdocs.yml
git add scripts/
git add MLFLOW_CLEANUP_SUMMARY.md
echo "✅ Changes staged"
echo ""

# Step 4: Commit
echo "Step 4: Committing changes..."
git commit -m "chore: remove mlflow.db from version control and enhance artifact management

- Remove mlflow.db from git tracking (gitignored, generated at runtime)
- Consolidate .gitignore MLflow entries (remove duplicates)
- Add comprehensive MLflow artifact patterns (*.db-shm, *.db-wal, .mlflow/)
- Enhance 'make clean' to remove MLflow artifacts  
- Add MLflow tracking documentation (docs/getting-started/mlflow-tracking.md)
- Update setup docs to clarify MLflow is optional
- Create cleanup scripts with dry-run mode
- Add verification and  commit helper scripts
- Update mkdocs navigation to include MLflow guide
- Add implementation summary (MLFLOW_CLEANUP_SUMMARY.md)

MLflow is now fully optional. Local DB and artifacts are generated at runtime
and never committed. Use 'make clean' or './scripts/clean_artifacts.sh' to
remove local tracking artifacts.

See: MLFLOW_CLEANUP_SUMMARY.md for details"

echo "✅ Changes committed"
echo ""

# Step 5: Verify
echo "Step 5: Running verification..."
./scripts/verify_mlflow_cleanup.sh
echo ""

echo "✅ All done! Ready to: git push origin main"
