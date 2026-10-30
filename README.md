# Cechy liści ziemniaka

`src/features` zamienia obrazy w macierz cech dla klasyfikatora.
Nie segmentuje liści, nie zmienia obrazów ani nie trenuje modeli.

## Instalacja

```bash
uv sync --locked
```

Domyślna instalacja nie pobiera stosu GPU ani Jupytera. Dodatki do
preprocessingu, treningu i notebooków opisuje [lista zależności](docs/dependencies.md).
API preprocessingu jest w [src/enhancement](src/enhancement/README.md).

## API dla pipeline

Obraz: RGB `(H, W, 3)`, NumPy `uint8` 0–255 lub float 0–1.
Obecny preprocessing zwraca obrazy PIL. Konwertujemy je na granicy modułów:

```python
import numpy as np
from src.features import extract_features

features = extract_features(np.asarray(image.convert("RGB")))
```

W notebooku można zachować dotychczasowe obrazy i numeryczne etykiety:

```python
from src.features import ImageSample, build_feature_matrix

samples = (
    ImageSample(f"train/{i}", np.asarray(image.convert("RGB")), label)
    for i, (image, label) in enumerate(zip(X_train, y_train, strict=True))
)
X_features, metadata = build_feature_matrix(samples)
y = metadata["label"].to_numpy()
```

`X_features` to DataFrame zawierający wyłącznie cechy liczbowe. Kolejność
wierszy odpowiada `metadata` z kolumnami `sample_id`, `label`, `image_sha256`.
Etykiety tekstowe i całkowite, w tym `0`/`1`, zachowują swoje wartości.
Identyfikatory muszą być unikalne; najlepiej używać stałych identyfikatorów źródła.
Hash pikseli pozwala modułowi treningowemu grupować dokładne duplikaty.

## Maski i cechy

Opcjonalne `leaf_mask` i `lesion_mask` to tablice `bool` `(H, W)`, wyrównane
z obrazem. Maska liścia musi obejmować także brązowe fragmenty. Maska plam musi
zawierać się w masce liścia; pusta oznacza brak wykrytych plam. Brak maski
nie oznacza zdrowego liścia. Obecny `restore_images` nie zwraca masek.

| Grupa | Liczba | Cechy |
|---|---:|---|
| `color` | 6 | Średnia i populacyjne odchylenie standardowe kanałów Lab L, a, b |
| `glcm` | 12 | Kontrast, jednorodność, energia, korelacja; odległości 1, 2, 4 px; średnia po czterech kierunkach |
| `lbp` | 28 | Histogramy uniform: 8 sąsiadów przy promieniu 1, 16 przy promieniu 2 |
| `shape` | 4 | Mimośród, solidity, extent, kolistość maski liścia |
| `lesions` | 4 | Udział plam, ich liczba, największy i średni udział plamy w powierzchni liścia |

Domyślnie liczymy `color`, `glcm`, `lbp`: **46 cech**. Bez maski obejmują cały
kadr z tłem. `groups=FEATURE_GROUPS` włącza wszystkie **54 cechy** i wymaga obu
masek; `FEATURE_GROUPS` jest eksportowane z `src.features`. Brak wymaganej maski
powoduje błąd, bez dopisywania zer ani NaN.

GLCM używa 32 poziomów szarości i wyłącznie par pikseli wewnątrz maski.
LBP pomija niepełne sąsiedztwa; każdy histogram sumuje się do 1.
Odległości są w pikselach: preprocessing powinien utrzymywać porównywalną skalę.
Skalowanie cech, wybór grup i podział danych należą do modułu treningowego.

## Eksport z folderu

```bash
uv run python -m src.features "data/Task 1 - Simple" outputs/easy_raw
uv run python -m src.features "data/Task 2 - Hard" outputs/hard_raw
```

Powstają `X.csv`, `samples.csv` oraz `metadata.json` z parametrami i wersjami
bibliotek. Oba CSV mają identyczną kolejność wierszy; etykiety nie trafiają do X.
`outputs/` jest ignorowany przez Git.

Opcjonalne maski PNG odwzorowują układ `klasa/nazwa.png`, z wartościami 0/1
albo 0/255. Adapter nie wykonuje resize, segmentacji ani korekcji kolorów:

```bash
uv run python -m src.features "data/Task 1 - Simple" outputs/easy_masked \
  --masks maski_lisci --lesions maski_plam \
  --groups color glcm lbp shape lesions
```

## Weryfikacja

```bash
uv run python -m pytest tests/test_features.py tests/test_dataset.py -q
```

Test integracji z preprocessingiem używa `dry_run=True`; nie potwierdza
wykonania modelu SUPIR na GPU.

## Wizualizacje deskryptorów

```bash
uv run --extra visualization python -m src.features.visualize \
  outputs/easy_raw outputs/hard_raw --output outputs/visualizations
```

Powstają przestrzenie cech oraz plansze RGB, Lab, GLCM i LBP dla obu zadań.
Każdy punkt to obraz; rzut PCA pokazuje tylko część informacji z pełnej macierzy.
Mapy przedstawiają obliczenia cech, a nie lokalizację choroby.

Podział należy do modułu treningowego. Opcjonalny `split.csv` przy macierzy
musi mieć identyczną kolejność `sample_id` i kolumnę `split` z wartościami
`development`/`test`. Wtedy wykresy pomijają test. Bez tego pliku pokazują
wszystkie dostarczone przykłady i jawnie oznaczają brak podziału.

Obecne podglądy wymagają `input_scope=whole_image`: nie ignorują po cichu
masek z przyszłego preprocessingu. `manifest.json` zapisuje zakres, przykłady
i zachowaną wariancję PCA. Wizualizacje nie trenują klasyfikatora.

```bash
uv run --extra visualization python -m pytest tests/test_visualize.py -q
```
