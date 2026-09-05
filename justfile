# Default recipe to list all available commands
default:
    @just --list

# Run all quality gates (Lint, Format, Types, Security, Tests)
check-all: lint format-check type-check security test

# Run the pytest suite with code coverage tracking
test:
    pytest

# Run ruff linter and automatically fix safe code violations
lint:
    ruff check . --fix

# 🔍 Check formatting rules without changing files
format-check:
    ruff format --check .

# Automatically format all source files using ruff
format:
    ruff format .

# Run static type checking across the package and its entry points
type-check:
    mypy src/ app/ scripts/

# Scan shipped code for security issues (Ruff's `S` rules run in `lint` too)
security:
    bandit -c pyproject.toml -q -r src/ app/ scripts/

# Launch the chat interface (needs the `ui` extra, a database, and a model service)
ui:
    chainlit run app/main.py

# Answer one question from the terminal, without the interface
ask query:
    python scripts/answer.py "{{query}}"

# Clean up temporary cache directories and build artifacts
clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build src/*.egg-info .files
