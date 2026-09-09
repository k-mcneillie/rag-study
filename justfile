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
    mypy src/ service/ app/src/ app/tests/ app-openai/src/ app-openai/tests/ scripts/

# Scan shipped code for security issues (Ruff's `S` rules run in `lint` too)
security:
    bandit -c pyproject.toml -q -r src/ service/ app/src/ app-openai/src/ scripts/

# Run the HTTP API service (needs the `api` extra, a database, and model assets)
serve:
    uvicorn service.app:create_app --factory \
        --host ${RAG_API_HOST:-127.0.0.1} --port ${RAG_API_PORT:-8080}

# Launch the chat interface (needs the `ui` extra and `pip install -e app/`;
# talks to the API service — run `just serve` first, or it starts in demo mode)
ui:
    chainlit run app/src/rag_chat/main.py

# Launch the OpenAI-dialect chat interface (needs `pip install -e app-openai/`;
# talks to an OpenAI-language provider — set RAG_OPENAI_BASE_URL, or it starts
# in demo mode)
ui-openai:
    chainlit run app-openai/src/rag_chat_openai/main.py

# Answer one question from the terminal, without the interface
ask query:
    python scripts/answer.py "{{query}}"

# Clean up temporary cache directories and build artifacts
clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build src/*.egg-info app/src/*.egg-info app-openai/src/*.egg-info .files
