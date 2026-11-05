---
theme: default
title: Rozpoznawanie chorób liści ziemniaka
info: Klasyczne metody rozpoznawania wzorców — mini-projekt i ćwiczenia 1–2
author: ''
lang: pl
colorSchema: light
aspectRatio: 16/9
canvasWidth: 1200
drawings:
  persist: false
fonts:
  provider: none
  sans: Archivo Variable
  mono: Archivo Variable
defaults:
  layout: default
transition: fade
duration: 7min40sec
presenter: true
download: false
class: cover
---

<div class="cover-copy">
  <h1>Rozpoznawanie<br>chorób liści<br>ziemniaka</h1>
  <p>Kolor, tekstura i kształt<br>w klasycznym uczeniu maszynowym</p>
</div>
<img class="cover-image" src="/images/cover.jpg" alt="Liść ziemniaka z widocznymi zmianami chorobowymi ze zbioru projektu">

<!--
Czas: 20 s.

W tym projekcie rozpoznajemy choroby liści ziemniaka na podstawie zdjęć. Zamiast trenować sieć neuronową, opisujemy obraz przez kolor, teksturę i kształt, a następnie przekazujemy te cechy klasycznym modelom. Najpierw krótko pokażę, jakie metody można było wykorzystać w dwóch wcześniejszych ćwiczeniach. Później przejdę przez nasz pipeline, porównanie modeli i jeden rzeczywisty błąd.

Źródła: ../model_training_simple.ipynb i ../model_training_hard.ipynb; pochodzenie zdjęcia: sources.json.
-->

---

# Lab 1. Klasyfikacja według kształtu

<div class="lab-layout">
  <figure class="shape-figure">
    <img src="/images/shape-mask.png" alt="Binarna maska liścia ilustrująca opis kształtu">
    <figcaption>Maska liścia jako ilustracja metody</figcaption>
  </figure>
  <div class="method-list">
    <div><ph-ruler-bold /><p><strong>Pole, obwód, zwartość</strong><span>Prosty opis sylwetki</span></p></div>
    <div><ph-wave-sine-bold /><p><strong>Deskryptory Fouriera</strong><span>Opis konturu</span></p></div>
    <div><ph-crosshair-bold /><p><strong>Centrowanie i skalowanie</strong><span>Porównanie wariantów + kNN / SVM</span></p></div>
  </div>
</div>
<p class="footnote">Możliwe rozwiązanie ćwiczenia; bez głębokich sieci</p>

<!--
Czas: 35 s.

Pierwsze ćwiczenie dotyczyło klasyfikacji obiektów według kształtu. Najprostszy wariant mógł wykorzystywać pole, obwód, wydłużenie i zwartość. Bardziej zaawansowanym rozwiązaniem są deskryptory Fouriera: opisujemy nimi przebieg konturu, zamiast analizować każdy piksel. Potem normalizujemy cechy i stosujemy na przykład kNN albo SVM. Instrukcja wymagała również porównania wpływu centrowania i skalowania oraz połączenia cech prostych z zaawansowanymi. Pokazana maska pochodzi z mini-projektu i tylko ilustruje tę ideę. Nie przedstawiamy jej jako wyniku labu pierwszego. Modele należało oceniać na wydzielonych danych walidacyjnych i testowych.

Źródło: ../../../lab1/Ćwiczenie 1 - klasyfikacja według kształtu.pdf, s. 1–2. Obraz: sources.json.
-->

---

# Lab 2. Klasyfikacja według tekstur

<div class="texture-layout">
  <div class="texture-pair">
    <figure><img src="/images/texture-healthy.png" alt="Fragment tekstury zdrowego liścia w skali szarości"><figcaption>Gładsza powierzchnia</figcaption></figure>
    <figure><img src="/images/texture-diseased.png" alt="Fragment tekstury liścia z plamami w skali szarości"><figcaption>Nieregularne plamy</figcaption></figure>
  </div>
  <div class="method-list">
    <div><ph-grid-four-bold /><p><strong>GLCM + kNN / SVM</strong><span>Kontrast, energia, jednorodność</span></p></div>
    <div><ph-scan-bold /><p><strong>Przesuwne okno</strong><span>Propozycja segmentacji tekstur</span></p></div>
    <div><ph-arrows-left-right-bold /><p><strong>Porównanie z GLRLM</strong><span>Długości serii podobnych pikseli</span></p></div>
  </div>
</div>
<p class="footnote">Możliwe rozwiązanie dla 4–5 klas; zdjęcia liści ilustrują różnicę tekstur</p>

<!--
Czas: 35 s.

Drugie ćwiczenie przenosiło uwagę z konturu na strukturę powierzchni. Dla czterech lub pięciu klas obrazów w skali szarości można policzyć macierze GLCM. Zliczają one pary jasności pikseli w zadanym sąsiedztwie. Otrzymujemy z nich na przykład kontrast, energię i jednorodność, a potem uczymy klasyfikator i mierzymy accuracy. Obliczanie takich cech w przesuwnym oknie daje propozycję segmentacji tekstur. Do porównania służy GLRLM, która opisuje długości serii podobnych pikseli. Instrukcja zawierała też dyskusję o CNN. Tutaj zdjęcia liści są ilustracją, a nie wynikami tego ćwiczenia.

Źródło: ../../../lab2/Ćwiczenie 2 - klasyfikacja według tekstur.pdf, s. 1. Obrazy: sources.json.
-->

---

# Dwa zadania rozpoznawania liści

<div class="class-examples">
  <figure><img src="/images/healthy.jpg" alt="Przykład zdrowego liścia ziemniaka"><figcaption>Zdrowy</figcaption></figure>
  <figure><img src="/images/early.jpg" alt="Liść ziemniaka z etykietą Early blight"><figcaption>Alternarioza<span>Early blight</span></figcaption></figure>
  <figure><img src="/images/late.jpg" alt="Liść ziemniaka z etykietą Late blight"><figcaption>Zaraza ziemniaka<span>Late blight</span></figcaption></figure>
</div>
<div class="task-summary">
  <p><strong>Proste · 305 zdjęć</strong><span>Zdrowy / alternarioza</span></p>
  <p><strong>Trudne · 2000 zdjęć</strong><span>Alternarioza / zaraza ziemniaka</span></p>
</div>

<!--
Czas: 35 s.

Mini-projekt łączy te pomysły, ale dodaje informację o kolorze. Mamy dwa osobne zadania binarne. W prostym rozróżniamy zdrowe liście i alternariozę: 152 zdjęcia zdrowych i 153 chorych liści. W trudnym obie klasy są chore: po tysiąc zdjęć alternariozy i zarazy ziemniaka. To ważne rozróżnienie, bo wysoki wynik pierwszego zadania nie oznacza jeszcze rozpoznawania konkretnej choroby. Przykłady pokazują też, że plamy mają różną wielkość i barwę. Traktujemy nazwy klas jako etykiety dostarczonego zbioru, a nie niezależnie potwierdzone diagnozy. Każde zadanie trenujemy i oceniamy osobno.

Źródła: oba notebooki treningowe, komórki 1 i 3; lokalne foldery data/Task 1 - Simple i data/Task 2 - Hard. Obrazy: sources.json.
-->

---

# Pipeline użyty w eksperymentach

<div class="pipeline">
  <div><ph-image-bold /><strong>Zdjęcie RGB</strong><span>Wejście</span></div>
  <ph-arrow-right-bold class="connector" />
  <div><ph-leaf-bold /><strong>Maska liścia</strong><span>Usunięcie tła</span></div>
  <ph-arrow-right-bold class="connector" />
  <div><ph-sliders-horizontal-bold /><strong>50 cech</strong><span>Kolor + tekstura + kształt</span></div>
  <ph-arrow-right-bold class="connector" />
  <div><ph-chart-scatter-bold /><strong>Klasyfikator</strong><span>StandardScaler + model</span></div>
</div>
<div class="variant-line">
  <p>Porównujemy trzy warianty obrazu</p>
  <strong>Sama maska <span>/</span> + min–max <span>/</span> + Gaussian, CLAHE i min–max</strong>
</div>
<p class="footnote">Maska powstaje na oryginale i pozostaje taka sama we wszystkich wariantach</p>

<!--
Czas: 45 s.

Najpierw z oryginalnego zdjęcia wyznaczamy maskę liścia. Poza nią tło staje się czarne, a ekstraktor liczy cechy wyłącznie w obrębie maski. Porównujemy trzy warianty: samą maskę, maskę z normalizacją min–max oraz wariant z filtrem Gaussa, CLAHE i normalizacją. CLAHE poprawia lokalny kontrast kanału jasności. Z każdego obrazu powstaje wektor pięćdziesięciu liczb. Następnie StandardScaler skaluje poszczególne cechy przed klasyfikatorem. Porównujemy SVM z jądrem RBF i liniowym, kNN, Random Forest oraz regresję logistyczną. To pipeline rzeczywiście wykorzystany w zapisanych eksperymentach. Obecny w repozytorium moduł SUPIR nie bierze udziału w tym treningu.

Źródła: ../src/training/data.py, VARIANTS, FEATURE_GROUPS, load_and_preprocess; ../src/training/experiment.py, MODELS, make_pipeline. CLAHE i filtr: ../src/preprocessing/adjustment.py i filtering.py.
-->

---

# Typowy liść: oddzielenie od tła

<div class="processing-triptych">
  <figure><img src="/images/typical-original.jpg" alt="Oryginalne zdjęcie zdrowego liścia"><figcaption>Oryginał</figcaption></figure>
  <figure><img src="/images/typical-mask.png" alt="Maska liścia wyliczona przez preprocessing projektu"><figcaption>Maska</figcaption></figure>
  <figure><img src="/images/typical-processed.png" alt="Liść po masce, filtrze Gaussa, CLAHE i normalizacji"><figcaption>Po przetworzeniu</figcaption></figure>
</div>
<p class="takeaway">Otsu + morfologia + otoczka wypukła największego konturu</p>

<!--
Czas: 40 s.

Na tym typowym zdrowym liściu widać cały proces. Maska łączy dwa sygnały: nasycenie w HSV oraz kanał a przestrzeni Lab, związany z osią zielony–czerwony. Progi dobiera metoda Otsu. Otwarcie i zamknięcie usuwają drobne zakłócenia, po czym wybieramy największy kontur i wypełniamy jego otoczkę wypukłą. Ten ostatni krok ma zachować również brązowe fragmenty przy krawędzi chorego liścia. Może jednak dołączyć tło we wcięciach konturu. Po prawej pokazujemy wariant z filtrem i CLAHE. To rzeczywisty wynik obecnego preprocessingu dla tego zdjęcia, bez ręcznej poprawy maski.

Źródła: ../src/preprocessing/morphology.py, create_leaf_mask; ../src/training/data.py, load_and_preprocess. Obrazy i parametry: sources.json.
-->

---

# Co opisuje 50 cech?

<div class="feature-groups">
  <div><img src="/images/descriptor-color.png" alt="Kanał Lab a zdrowego liścia, skala od minus 25 do 25"><strong>Kolor Lab</strong><b>6</b><p>Średnie i odchylenia</p></div>
  <div><img class="pixelated" src="/images/descriptor-glcm.png" alt="Macierz współwystępowania jasności, 32 poziomy szarości"><strong>Tekstura GLCM</strong><b>12</b><p>Pary jasności pikseli</p></div>
  <div><img src="/images/descriptor-lbp.png" alt="Mapa kodów uniform LBP dla promienia 1"><strong>Tekstura LBP</strong><b>28</b><p>Lokalne wzorce</p></div>
  <div><img src="/images/typical-mask.png" alt="Maska liścia do wyliczania cech kształtu"><strong>Kształt</strong><b>4</b><p>Geometria maski</p></div>
</div>
<p class="takeaway">W zapisanej analizie największy wpływ ma kolor</p>
<p class="footnote">Ważność: spadek F1 macro po permutacji grupy cech na teście</p>

<!--
Czas: 40 s.

Wektor wejściowy ma cztery grupy. Miniatury pokazują obliczenia; kolory map nie oznaczają lokalizacji choroby. Kolor to średnia i odchylenie standardowe każdego z trzech kanałów Lab. GLCM opisuje relacje jasności sąsiednich pikseli dla kilku odległości. LBP zlicza lokalne wzorce w dwóch skalach. Cztery cechy kształtu opisują maskę liścia. Nie wykorzystujemy cech pojedynczych plam, ponieważ ten pipeline ich nie segmentuje. W notebookach po przetasowaniu cech koloru jakość spada najmocniej, a kształt wnosi niewiele. To analiza wrażliwości danego modelu na tym zbiorze. Nie dowodzi, że kolor wystarczy w innych warunkach, ani że model rozpoznaje konkretne zmiany chorobowe na zdjęciu.

Źródła: ../src/features/extract.py; ../src/training/data.py, FEATURE_GROUPS; oba notebooki, komórki 4 i 17 (pierwszy wykres); ../src/training/experiment.py, feature_importance.
-->

---
class: dark-slide
---

# Jak ocenialiśmy modele?

<div class="split-visual">
  <div class="development"><strong>Uczenie · około 70%</strong><span>5-fold CV + dobór parametrów</span></div>
  <div class="test"><strong>Test · około 30%</strong><span>Ocena po wyborze konfiguracji</span></div>
</div>
<div class="split-counts"><p>Proste: <strong>212 / 93</strong></p><p>Trudne: <strong>1400 / 600</strong></p></div>
<p class="takeaway">Wybór konfiguracji według F1 macro w CV</p>
<p class="footnote">Grupowanie chroni przed identycznymi obrazami; różne zdjęcia tego samego liścia mogą trafić do obu części</p>

<!--
Czas: 45 s.

Około siedemdziesięciu procent zdjęć służy do uczenia, a trzydzieści do testu. W części uczącej wykonujemy pięciokrotną walidację krzyżową i wybieramy parametry wraz z wariantem preprocessingu według F1 macro. Ta miara nadaje obu klasom równą wagę. StandardScaler jest dopasowywany wewnątrz każdego folda, więc nie korzysta z danych walidacyjnych. Obrazy identyczne po usunięciu tła grupujemy po hashu pikseli; w tych zbiorach nie wykryto takich duplikatów. To zabezpieczenie nie rozpoznaje jednak różnych zdjęć tego samego liścia. Podział nie gwarantuje więc niezależności biologicznych obiektów, a wyniki mogą być optymistyczne. Nie mamy też oddzielnego testu terenowego.

Źródła: ../src/training/experiment.py, group_train_test_split, make_split, run_search; oba notebooki, komórki 7, 8 i 20. Grupowanie obejmuje hash obrazu po wariancie bazowym, nie identyfikator liścia.
-->

---

# Zysk z preprocessingu zależy od modelu

<table class="preprocessing-table">
  <caption>F1 macro w CV · sama maska i wariant Gaussian + CLAHE + min–max</caption>
  <thead><tr><th></th><th>Sama maska</th><th>Pełny wariant</th><th>Zmiana</th></tr></thead>
  <tbody>
    <tr><th>SVM RBF · proste</th><td>0,9811</td><td>0,9906</td><td class="positive">+0,94 pp</td></tr>
    <tr><th>SVM RBF · trudne</th><td>0,9714</td><td>0,9721</td><td class="positive">+0,07 pp</td></tr>
    <tr><th>kNN · proste</th><td>0,9573</td><td>0,9196</td><td class="negative">−3,78 pp</td></tr>
    <tr><th>kNN · trudne</th><td>0,9421</td><td>0,9235</td><td class="negative">−1,86 pp</td></tr>
  </tbody>
</table>
<p class="takeaway">W zadaniu trudnym poprawa SVM jest niewielka</p>

<!--
Czas: 40 s.

Ten slajd pokazuje wyniki walidacji, a nie testu. W prostym zadaniu dodatkowe przetwarzanie poprawia najlepszy wynik SVM RBF o niecały punkt procentowy. W trudnym zysk wynosi zaledwie siedem setnych punktu procentowego, więc jest mały wobec zmienności między foldami. Dla kNN ten sam pakiet operacji pogarsza wynik w obu zadaniach. Nie można zatem powiedzieć, że więcej preprocessingu zawsze pomaga. Nie izolowaliśmy też wpływu samego CLAHE: zmieniały się jednocześnie filtracja i normalizacja, a parametry modeli dobierano osobno. Tabela porównuje najlepsze konfiguracje dostępne w danym wariancie obrazu.

Źródła: oba notebooki, komórki 9 i 11; różnice obliczone z pełnych wartości komórki 9. Zmiany w punktach procentowych policzono przed zaokrągleniem wartości F1.
-->

---

# Wyniki na zbiorze testowym

<ResultsChart />
<p class="takeaway">SVM wygrywa w CV; regresja logistyczna ma o 1 trafienie więcej na trudnym teście</p>

<!--
Czas: 50 s.

Tutaj porównujemy accuracy na teście dla konfiguracji każdego modelu wybranej wcześniej w CV. W prostym zadaniu oba SVM poprawnie klasyfikują 91 z 93 zdjęć, czyli 97,8 procent. Dla SVM RBF obie pomyłki to chore liście uznane za zdrowe. W trudnym zadaniu SVM RBF trafia 580 z 600 zdjęć, czyli 96,7 procent. Regresja logistyczna uzyskuje 581 trafień, o jedno więcej. Wybór SVM nadal wynika z walidacji, a nie z tego testu. Różnica jednego zdjęcia nie wystarcza do mocnego wniosku o przewadze któregoś modelu. kNN ma najniższą accuracy w obu zadaniach. Dla zwycięskiego SVM testowe F1 macro wynosi odpowiednio około 0,978 i 0,967.

Źródła: oba notebooki, komórki 9, 13 i 15. Wartości procentowe wykresu wyliczono z całkowitych liczebności zapisanych macierzy pomyłek. Oś wykresu jest celowo ograniczona do 90–100%, jak wskazuje podpis.
-->

---

# Trudny przypadek: pomyłka wszystkich modeli

<div class="error-layout">
  <figure><img src="/images/tough-original.jpg" alt="Trudny przypadek Early blight błędnie rozpoznany przez wszystkie pięć modeli"><figcaption>Etykieta: <strong>alternarioza</strong></figcaption></figure>
  <figure><img src="/images/tough-processed.png" alt="Ten sam trudny liść po przetworzeniu wariantem wybranego SVM"><figcaption>Predykcja: <strong>zaraza ziemniaka</strong></figcaption></figure>
  <div class="confusion">
    <p>SVM RBF · trudny test</p>
    <table aria-label="Macierz pomyłek SVM RBF. Wiersze: etykiety, kolumny: predykcje.">
      <thead><tr><th></th><th>Alt.</th><th>Zar.</th></tr></thead>
      <tbody><tr><th>Alt.</th><td>291</td><td class="mistake">9</td></tr><tr><th>Zar.</th><td class="mistake">11</td><td>289</td></tr></tbody>
    </table>
    <span>Wiersze: etykiety<br>Kolumny: predykcje</span>
  </div>
</div>
<p class="takeaway">6 zdjęć w trudnym teście pomyliło wszystkie 5 modeli</p>

<!--
Czas: 45 s.

To konkretny przykład z zapisanej analizy błędów. Zdjęcie ma etykietę alternariozy, ale wszystkie pięć modeli wskazało zarazę ziemniaka. Po prawej od oryginału widać obraz po przetworzeniu wariantem wybranego SVM. Widoczna maska i podobne plamy nie pozwalają nam jednoznacznie ustalić przyczyny błędu. Możliwe wyjaśnienia to nakładanie się cech klas albo ograniczenia segmentacji; nie mamy ich tutaj potwierdzonych. Macierz obok dotyczy całego trudnego testu: dziewięć alternarioz rozpoznano jako zarazę, a jedenaście zaraz jako alternariozę. Sześć zdjęć sprawiło problem wszystkim modelom. Wysoki wynik zbiorczy nadal pozostawia takie przypadki.

Źródło: ../model_training_hard.ipynb, komórki 15 i 19. Identyfikator przykładu i pochodzenie obrazów: sources.json. Etykieta Early blight = 0, Late blight = 1; kolumny macierzy są predykcjami. Nie jest to potwierdzona diagnoza ani potwierdzona przyczyna pomyłki.
-->

---
class: dark-slide conclusion
---

# Wnioski i dalsze kroki

<div class="conclusion-lines">
  <p><ph-check-circle-bold /><span>Klasyczne modele dobrze rozdzielają klasy <strong>w tym zbiorze</strong></span></p>
  <p><ph-palette-bold /><span>Najwięcej informacji wnosi <strong>kolor</strong></span></p>
</div>

<!--
Czas: 30 s.

Podsumowując, połączenie maski liścia, ręcznie zaprojektowanych cech i klasycznych modeli daje wysokie wyniki na dostarczonym zbiorze. Najważniejszy okazuje się kolor, a dodatkowe przetwarzanie nie daje uniwersalnej poprawy. Następnym krokiem byłby test, w którym żaden liść ani seria jego zdjęć nie występuje wcześniej w uczeniu, oraz sprawdzenie fotografii terenowych. Można też zbadać segmentację plam. To propozycja dalszego eksperymentu. Obecne pomiary nie wystarczają do oceny narzędzia diagnostycznego w praktyce.

Źródła: oba notebooki, komórki 17–20; ../src/training/data.py, FEATURE_GROUPS. Nowe testy i segmentacja plam są propozycjami dalszych prac.
-->
