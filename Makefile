push:
	@git add .
	@git commit -m "$(m)"
	@git push

# Usage: make push msg="your commit message"

.PHONY: push