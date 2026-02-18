#!/usr/bin/env bash
# Clean up MLflow and experiment artifacts
# Usage: ./scripts/clean_artifacts.sh [--dry-run]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Clean up MLflow artifacts and experiment outputs."
            echo ""
            echo"Options:"
            echo "  --dry-run    Show what would be deleted without actually deleting"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}🧹 Cleaning MLflow and experiment artifacts...${NC}"
echo ""

# Function to remove files/directories
remove_item() {
    local item=$1
    if [ -e "$item" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "  ${YELLOW}[DRY RUN]${NC} Would remove: $item"
        else
            rm -rf "$item"
            echo -e "  ${GREEN}✓${NC} Removed: $item"
        fi
    else
        echo -e "  ${GREEN}✓${NC} Not found (already clean): $item"
    fi
}

# MLflow artifacts
echo -e "${YELLOW}MLflow artifacts:${NC}"
remove_item "mlflow.db"
remove_item "mlflow.db-shm"
remove_item "mlflow.db-wal"
remove_item "mlruns/"
remove_item "mlartifacts/"
remove_item ".mlflow/"

echo ""
echo -e "${YELLOW}Python cache:${NC}"
find . -type d -name "__pycache__" -print0 | while IFS= read -r -d '' dir; do
    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY RUN]${NC} Would remove: $dir"
    else
        rm -rf "$dir"
        echo -e "  ${GREEN}✓${NC} Removed: $dir"
    fi
done

find . -type f -name "*.pyc" -print0 | while IFS= read -r -d '' file; do
    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY RUN]${NC} Would remove: $file"
    else
        rm -f "$file"
        echo -e "  ${GREEN}✓${NC} Removed: $file"
    fi
done

echo ""
echo -e "${YELLOW}Test and coverage artifacts:${NC}"
remove_item ".pytest_cache/"
remove_item ".coverage"
remove_item "htmlcov/"

echo ""
echo -e "${YELLOW}Build artifacts:${NC}"
remove_item "build/"
remove_item "dist/"
find . -type d -name "*.egg-info" -print0 | while IFS= read -r -d '' dir; do
    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY RUN]${NC} Would remove: $dir"
    else
        rm -rf "$dir"
        echo -e "  ${GREEN}✓${NC} Removed: $dir"
    fi
done

echo ""
echo -e "${YELLOW}Linter caches:${NC}"
remove_item ".ruff_cache/"
remove_item ".mypy_cache/"

echo ""
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}ℹ️  This was a dry run. Run without --dry-run to actually delete files.${NC}"
else
    echo -e "${GREEN}✅ Cleanup complete!${NC}"
    echo ""
    echo "Tip: Run 'make clean' for the same effect, or use './scripts/clean_artifacts.sh --dry-run' to preview."
fi
