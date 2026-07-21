# BlueprintAI local infrastructure
# Requires: Docker Desktop, native Ollama (brew install ollama) for GPU vision.

MODELS = llama3.2-vision:11b llama3.1:8b snowflake-arctic-embed:335m

.PHONY: up down build logs status models health clean

up: ## build and start the whole stack (db, minio, backend, frontend)
	docker compose up -d --build
	@echo "\nBlueprintAI is starting:"
	@echo "  app:      http://localhost:5175"
	@echo "  api:      http://localhost:8000/docs"
	@echo "  minio:    http://localhost:9001 (minioadmin/minioadmin)"
	@echo "\nFirst run? Pull the AI models once:  make models"

down: ## stop the stack (data volumes are kept)
	docker compose down

build: ## rebuild images without starting
	docker compose build

logs: ## follow backend logs
	docker compose logs -f backend

status: ## show service status + health
	docker compose ps

models: ## pull the local (American-based) AI models into native Ollama
	@which ollama >/dev/null || (echo "Ollama not installed: brew install ollama" && exit 1)
	@for m in $(MODELS); do echo "pulling $$m..."; ollama pull $$m; done

health: ## quick end-to-end health probe
	@curl -sf http://localhost:8000/health >/dev/null && echo "backend  OK" || echo "backend  DOWN"
	@curl -sf http://localhost:5175 >/dev/null && echo "frontend OK" || echo "frontend DOWN"
	@curl -sf http://localhost:11434/api/tags >/dev/null && echo "ollama   OK" || echo "ollama   DOWN (brew services start ollama)"

clean: ## stop and DELETE ALL DATA (db + storage volumes)
	docker compose down -v
