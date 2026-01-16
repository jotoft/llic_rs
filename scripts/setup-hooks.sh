#!/bin/bash
# Install git hooks for the repository

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
ln -sf "$SCRIPT_DIR/hooks/pre-commit" "$HOOKS_DIR/pre-commit"
echo "Installed pre-commit hook"

echo "Done!"