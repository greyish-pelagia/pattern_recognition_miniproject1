# Prezentacja: choroby liści ziemniaka

12 slajdów po polsku, planowany czas **7:40**. Notatki prelegenta są
w komentarzach w `slides.md`. Zawierają tekst wypowiedzi, czas i źródła.

## Uruchomienie

```bash
cd presentation
bun install --frozen-lockfile
bun run dev
```

Slajdy: <http://localhost:3030>. Tryb prelegenta z notatkami:
<http://localhost:3030/presenter>. Strzałki zmieniają slajdy.

```bash
bun run check   # struktura, czas, obrazy i ich źródła
bun run build   # samodzielna strona w dist/
bun run export  # aktualizacja slides.pdf
```

Eksport wymaga Chromium dla Playwright. Jeśli nie jest zainstalowany:
`bunx playwright install chromium`. Polecenia eksportu opisuje
[dokumentacja Slidev](https://sli.dev/guide/exporting).
Font Archivo jest dostarczany przez pakiet; slajdy nie pobierają Google Fonts.

## Plan wypowiedzi

| Slajd | Temat | Czas |
|---|---|---:|
| 1 | Wprowadzenie | 0:20 |
| 2 | Lab 1: możliwe metody opisu kształtu | 0:35 |
| 3 | Lab 2: możliwe metody opisu tekstury | 0:35 |
| 4 | Dane i dwa zadania | 0:35 |
| 5 | Pipeline | 0:45 |
| 6 | Typowy liść i preprocessing | 0:40 |
| 7 | Grupy cech | 0:40 |
| 8 | Podział danych i walidacja | 0:45 |
| 9 | Porównanie preprocessingu | 0:40 |
| 10 | Wyniki modeli | 0:50 |
| 11 | Rzeczywista pomyłka i macierz pomyłek | 0:45 |
| 12 | Wnioski | 0:30 |
| | **Łącznie** | **7:40** |

Czas zakłada około 125–135 słów na minutę oraz wskazywanie elementów na
slajdach. Warto przeprowadzić jedną próbę z własnym tempem mówienia.

## Dane i interpretacja

Wyniki pochodzą z **zapisanych wyjść** `model_training_simple.ipynb`
i `model_training_hard.ipynb`, z aktualnego repozytorium. Nie wykonywano
ponownego treningu. Użyty kod: `src/training/`, `src/preprocessing/`
i `src/features/extract.py`.

- Trening używa **50 cech z maską**: 6 Lab, 12 GLCM, 28 LBP, 4 kształtu.
  Domyślne 46 cech opisane w głównym README dotyczy innego wywołania API.
- SVM RBF wygrywa walidację w obu zadaniach. Na prostym teście oba SVM
  mają 91/93 poprawnych predykcji. Na trudnym teście SVM RBF ma 580/600,
  a regresja logistyczna 581/600. Jeden obraz nie dowodzi przewagi modelu.
- Zmiany F1 w slajdzie 9 obliczono z pełnych wartości komórki 9,
  przed zaokrągleniem liczb do wyświetlenia.
- Macierz na slajdzie 11 dotyczy SVM RBF i całego trudnego testu.
  Przykład błędu pochodzi z pierwszej pary obrazów w komórce 19.
- Grupowanie po hashu chroni przed identycznymi obrazami, ale nie zapewnia
  rozdzielenia różnych zdjęć tego samego liścia. Brakuje testu terenowego.
- SUPIR ani segmentacja plam nie uczestniczą w pokazanym treningu.

Slajdy o lab1 i lab2 pokazują **możliwe rozwiązania**, na podstawie lokalnych
instrukcji PDF. Nie przypisują studentom wykonanych eksperymentów.
Sylwetka i tekstury z liści są ilustracją metod.

`sources.json` opisuje źródło każdego obrazu, hash oryginału i przetwarzanie.
Zdjęcia są rzeczywistymi przykładami ze zbioru projektu. Maski i obrazy
po przetworzeniu wyliczono aktualnym kodem; float przekonwertowano na uint8
wyłącznie do wyświetlenia. Mapy Lab, GLCM i LBP ilustrują deskryptory,
nie lokalizację choroby. Zdjęcie typowe nie ma przypisanej predykcji.

Źródłowe instrukcje ćwiczeń są w katalogach `lab1/` i `lab2/`, poza tym
repozytorium. Ich ścieżki zapisano w notatkach odpowiednich slajdów.
Obsługę notatek opisuje [dokumentacja Slidev](https://sli.dev/guide/ui).
