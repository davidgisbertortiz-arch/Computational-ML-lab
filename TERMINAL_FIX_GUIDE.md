# Terminal Error Fix - Quick Guide

## What Happened

The terminal command got corrupted/concatenated, causing Exit Code 2. The commit message was garbled.

## How to Fix (Choose One Method)

### Method 1: Automated Script (Recommended)

```bash
# Make the fix script executable
chmod +x scripts/fix_and_commit.sh

# Run it
./scripts/fix_and_commit.sh
```

This will:
1. Remove mlflow.db from git
2. Make all scripts executable
3. Stage all changes
4. Commit with proper message
5. Run verification

---

### Method 2: Manual Steps

Run these commands one at a time:

```bash
# 1. Remove mlflow.db from git tracking
git rm --cached mlflow.db

# 2. Make scripts executable  
chmod +x scripts/*.sh

# 3. Stage all changes
git add .gitignore Makefile README.md docs/ mkdocs.yml scripts/ MLFLOW_CLEANUP_SUMMARY.md

# 4. Commit with clean message
git commit -m "chore: remove mlflow.db and enhance artifact management

- Remove mlflow.db from git tracking (gitignored, generated at runtime)
- Consolidate .gitignore MLflow entries
- Add MLflow artifact patterns (*.db-shm, *.db-wal, .mlflow/)
- Enhance 'make clean' to remove MLflow artifacts
- Add MLflow documentation (docs/getting-started/mlflow-tracking.md)
- Update setup docs to clarify MLflow is optional
- Create cleanup and verification scripts
- Update mkdocs navigation

MLflow is now fully optional. Use 'make clean' to remove local artifacts.

See: MLFLOW_CLEANUP_SUMMARY.md"

# 5. Verify everything
./scripts/verify_mlflow_cleanup.sh
```

---

## Then Test

```bash
# Run tests
make test

# Run linting
make lint

# Verify docs build
mkdocs build
```

---

## Finally Push

```bash
git push origin main
```

---

## If You Need to Reset

If something went wrong and you need to start over:

```bash
# Unstage everything
git reset HEAD

# Start fresh with Method 1 or 2
```

---

## Quick Status Check

```bash
# See what's staged
git status

# See what will be committed
git diff --cached --stat

# Verify mlflow.db not tracked
git ls-files | grep mlflow
# Should return nothing
```
