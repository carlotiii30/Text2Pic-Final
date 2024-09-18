.DEFAULT_GOAL := help
.PHONY: help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run: ## Run server
	poetry run python -m main

train_image: ## Train model
	poetry run python -m src.images.run

train_number: ## Train model
	poetry run python -m src.nums.training