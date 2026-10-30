# Zależności

Domyślnie instalujemy tylko ekstrakcję cech oraz narzędzia deweloperskie:

```bash
uv sync --locked
```

Nie instaluje to Torch, Transformers, OpenCV ani Jupytera. Wybierz dodatki
potrzebne do swojej części projektu:

| Dodatek | Zawartość | Instalacja |
|---|---|---|
| `enhancement` | SUPIR: Torch, torchvision, OpenCLIP, Transformers i zależności | `uv sync --extra enhancement` |
| `training` | scikit-learn, OpenCV, Matplotlib | `uv sync --extra training` |
| `notebooks` | ipykernel | `uv sync --extra notebooks` |
| `visualization` | Matplotlib i scikit-learn do PCA | `uv sync --extra visualization` |

Dodatki można łączyć, np. `uv sync --extra training --extra notebooks`.
Przy `uv run` także podaj potrzebne dodatki: `uv run --extra enhancement ...`.
`uv sync --no-dev` pomija pytest i Ruff. `--all-extras` pobiera również stos ML.

Wymagania istniejącego preprocessingu i notebooków zachowano. Lockfile zawiera
również pakiety opcjonalne; ich obecność w pliku nie oznacza instalacji.
