# Python restoration implementation plan

**Goal:** Expose `restore_images(images: list[PIL.Image.Image]) -> list[PIL.Image.Image]`, with keyword checkpoint configuration, and remove application tooling.

**Architecture:** Keep SUPIR inference and its checkpoint parameter names. Use a fixed prompt, lazy model loading and a cached model; process mixed image sizes sequentially. Return one RGB image at the input size, preserving order. No image file I/O in the entrypoint.

**Scope:** Remove LLaVA (confirmed by user), CLI, GUI, serving, evaluation, training-only modules, alternate configurations and demo assets. Keep license and inference dependencies. Resolve internal imports within the package without global module aliases.

- [x] Add regression coverage for empty and invalid input, mixed sizes and modes, ordering, lazy loading, and package/config import resolution. Run before implementation.
- [x] Implement the entrypoint and explicit checkpoint configuration; retain CUDA inference behavior and fixed SUPIR prompts.
- [x] Prune unrelated code and dependencies, including training hooks; preserve inference module names and state dictionary keys.
- [x] Document the Python API and required weights. Run unit tests and syntax/import-target checks; report that GPU inference is unavailable locally.

Validation: nine unittest checks pass; all retained Python files compile. Read-only review found a model-switching memory issue, fixed with eviction before replacement and a regression test. Full CUDA inference was not run because PyTorch, CUDA and weights are unavailable in this environment.
