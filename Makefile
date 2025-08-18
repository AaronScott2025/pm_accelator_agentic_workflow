# AI Interviewer PM - Development Makefile
# Convenience targets for local development and deployment

.PHONY: help install clean qdrant-up qdrant-down lint format type test test-fast test-cov kernel serve compose-up compose-down

# Default target
help:
	@echo "🎯 AI Interviewer PM - Available Commands"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  install       Install all dependencies with Poetry"
	@echo "  clean         Clean up temporary files and caches"
	@echo ""
	@echo "🗄️  Database Management:"
	@echo "  qdrant-up     Start Qdrant vector database"
	@echo "  qdrant-down   Stop Qdrant vector database"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  lint          Run linting checks with Ruff"
	@echo "  format        Format code with Black"
	@echo "  type          Run type checking with MyPy"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test          Run full test suite"
	@echo "  test-fast     Run quick tests (skip external services)"
	@echo "  test-cov      Run tests with coverage report"
	@echo ""
	@echo "🚀 Development:"
	@echo "  serve         Start development server"
	@echo "  kernel        Install Jupyter kernel"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  compose-up    Start all services with Docker Compose"
	@echo "  compose-down  Stop all Docker Compose services"

# Setup & Installation
install:
	@echo "📦 Installing dependencies..."
	poetry install --with dev --all-extras
	@echo "✅ Installation complete!"

clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Database Management
qdrant-up:
	@echo "🗄️  Starting Qdrant vector database..."
	docker compose up -d qdrant
	@echo "✅ Qdrant is running at http://localhost:6333"

qdrant-down:
	@echo "🛑 Stopping Qdrant..."
	docker compose down
	@echo "✅ Qdrant stopped!"

# Code Quality
lint:
	@echo "🔍 Running linting checks..."
	poetry run ruff check .
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting code..."
	poetry run black .
	@echo "✅ Code formatted!"

type:
	@echo "🔍 Running type checks..."
	poetry run mypy .
	@echo "✅ Type checking complete!"

# Testing
test:
	@echo "🧪 Running full test suite..."
	poetry run pytest -v
	@echo "✅ All tests complete!"

test-fast:
	@echo "⚡ Running quick tests..."
	poetry run pytest -q -k "not graph and not vectorstore"
	@echo "✅ Quick tests complete!"

test-cov:
	@echo "📊 Running tests with coverage..."
	poetry run pytest --cov=ai_interviewer_pm --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

# Development
serve:
	@echo "🚀 Starting development server..."
	poetry run uvicorn ai_interviewer_pm.api.server:app --reload --port 8080

kernel:
	@echo "📓 Installing Jupyter kernel..."
	poetry run python -m ipykernel install --user --name ai-interviewer-pm --display-name "Python (ai-interviewer-pm)"
	@echo "✅ Jupyter kernel installed!"

# Docker
compose-up:
	@echo "🐳 Starting all services with Docker Compose..."
	docker compose up --build -d
	@echo "✅ All services are running!"
	@echo "🌐 Frontend: http://localhost:3000"
	@echo "🔧 API: http://localhost:8080"
	@echo "📚 API Docs: http://localhost:8080/docs"
	@echo "🗄️  Qdrant: http://localhost:6333"

compose-down:
	@echo "🛑 Stopping all Docker Compose services..."
	docker compose down
	@echo "✅ All services stopped!"

# Quality check combo
check: lint format type test-fast
	@echo "✅ All quality checks passed!"

# Full development setup
setup: install qdrant-up
	@echo "🎉 Development environment ready!"
	@echo "Run 'make serve' to start the development server"
