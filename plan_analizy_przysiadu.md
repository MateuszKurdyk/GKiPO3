# Plan projektu: analiza poprawności przysiadu z filmu

## 1. Cel projektu

Stworzyć skrypt, który analizuje nagranie wykonywania przysiadu i:

- wykrywa osobę na kolejnych klatkach filmu,
- wyznacza punkty kluczowe ciała,
- rozpoznaje fazy ruchu: pozycja początkowa, zejście, dół przysiadu i powrót,
- ocenia wybrane elementy techniki,
- pokazuje wynik na obrazie oraz zapisuje podsumowanie.

Projekt będzie realizował temat „Trener sportowy - detekcja poprawności wykonywania ćwiczeń sportowych”. Nie powinien zastępować trenera ani diagnozy medycznej; wynik będzie orientacyjną analizą nagrania.

## 2. Najważniejsze decyzje do omówienia

### 2.1. Jaki typ filmu obsługujemy?

Najprostszy zakres:

- jedna osoba,
- widok z boku,
- całe ciało widoczne w kadrze,
- nieruchoma kamera,
- dobre oświetlenie,
- ćwiczenie bez dodatkowych obciążeń albo z jednym obciążeniem.

Później można dodać widok z przodu, wiele osób, ruchomą kamerę i różne warianty przysiadu. Nie warto zaczynać od wszystkich przypadków jednocześnie, ponieważ zmiana perspektywy zmienia interpretację kątów i widoczność punktów ciała.

### 2.2. Jak definiujemy „poprawny” przysiad?

Do pierwszej wersji można przyjąć:

- kolano pozostaje mniej więcej w osi stopy,
- plecy nie pochylają się nadmiernie,
- zejście osiąga ustaloną głębokość,
- pięty pozostają na podłożu,
- ruch jest płynny,
- powrót kończy się wyprostowaniem bez gwałtownego szarpnięcia.

Progi kątów powinny być konfigurowalne, a nie wpisane na stałe. Różnice budowy ciała, mobilności i celu ćwiczenia oznaczają, że nie istnieje jeden uniwersalny zestaw progów.

## 3. Proponowane warianty techniczne

### Wariant A — punkty pozy i reguły geometryczne (rekomendowany na start)

Pipeline:

1. OpenCV odczytuje film klatka po klatce.
2. Preprocessing poprawia obraz.
3. Model pose-estimation wykrywa punkty: biodro, kolano, kostka, bark, pięta i stopa.
4. Skrypt oblicza kąty między odcinkami ciała.
5. Reguły klasyfikują fazę ruchu i błędy.
6. OpenCV rysuje szkielet, kąty, komunikaty i wynik.

Zalety:

- szybki do wykonania i wyjaśnienia,
- wynik można uzasadnić konkretnym kątem,
- nie wymaga dużego własnego zbioru danych,
- dobrze pasuje do wymogu użycia YOLO i CV2, jeśli użyjemy modelu YOLO Pose.

Wady:

- wrażliwość na zasłonięte punkty i słabą perspektywę,
- reguły wymagają kalibracji,
- ocena techniki będzie uproszczona.

### Wariant B — YOLO do osoby/obiektu + model pozy

YOLO wykrywa osobę lub sztangę, a osobny model pozy wyznacza punkty ciała. OpenCV odpowiada za film, preprocessing i wizualizację.

Zalety:

- można kontrolować, czy analizowany jest właściwy człowiek,
- łatwiej później dodać sztangę, hantle albo liczenie powtórzeń,
- architektura jest modularna.

Wady:

- więcej elementów do skonfigurowania,
- większe wymagania obliczeniowe,
- detekcja obiektu sama nie mówi, czy przysiad jest poprawny.

### Wariant C — klasyfikator uczony na całych klatkach lub sekwencjach

Model otrzymuje przykłady poprawnych i niepoprawnych przysiadów oraz uczy się przewidywać klasę.

Zalety:

- może nauczyć się bardziej złożonych zależności,
- potencjalnie daje prostą klasyfikację końcową.

Wady:

- potrzebny jest opisany zbiór filmów,
- trudniej wyjaśnić, dlaczego model uznał ruch za błędny,
- model może nauczyć się tła, ubrania lub pozycji kamery zamiast techniki,
- większe ryzyko problemów z generalizacją.

**Decyzja robocza:** rozpocząć od wariantu A, a wariant B potraktować jako rozszerzenie. Wariant C zostawić jako możliwy etap eksperymentalny.

## 3.1. Ustalony zakres pierwszej wersji

- kamera ustawiona z boku,
- jedna osoba wykonująca przysiad ze sztangą,
- środowisko: GitHub Codespaces,
- wynik: film z wizualizacją oraz raport JSON,
- wykrywane problemy: zbyt mała głębokość, nadmierne pochylenie tułowia, kolana uciekające do środka i odrywanie pięt.

Ocena kolan uciekających do środka z widoku bocznego będzie ograniczona. Ten błąd jest zwykle łatwiejszy do oceny z przodu lub z tyłu, dlatego skrypt powinien oznaczać tę ocenę niższą pewnością albo traktować ją jako funkcję eksperymentalną. Jeśli będzie to kluczowy element projektu, później warto dodać drugi widok kamery.

### Dodatkowe ustalenia

- wejściem będzie gotowy plik wideo, np. MP4 lub MOV,
- kamera będzie nieruchoma, a całe ciało i sztanga mają pozostać w kadrze,
- komunikaty o błędach będą wyświetlane bezpośrednio na analizowanym filmie,
- projekt nie będzie wyliczał arbitralnej oceny punktowej; JSON zapisze pomiary, fazy ruchu i wykryte błędy,
- pierwsza wersja użyje gotowego modelu YOLO Pose, bez własnego trenowania,
- film może zawierać wiele powtórzeń,
- definicja głębokości będzie oparta na relacji biodra względem kolana,
- chwilowe naruszenie reguły nie będzie od razu zgłaszane; błąd musi utrzymać się przez kilka kolejnych klatek,
- klatki z niepewną detekcją zostaną pominięte i nie będą automatycznie uznawane za błędy,
- pliki będą analizowane lokalnie, bez wysyłania nagrań do zewnętrznych usług.

### Domyślne parametry nagrania telefonem

Przyjmujemy następujące wymagania dla materiału wejściowego:

- telefon ustawiony poziomo, najlepiej na statywie,
- nagranie 1080p przy 30 FPS; 4K nie jest wymagane,
- kamera z boku, prostopadle do płaszczyzny ruchu,
- odległość około 3-4 metrów, tak aby całe ciało, stopy i sztanga były widoczne z zapasem,
- nieruchoma kamera bez zoomu cyfrowego,
- osoba i sztanga dobrze oświetlone, bez silnego światła za plecami,
- przed rozpoczęciem serii około 2 sekund stania nieruchomo,
- po zakończeniu serii około 2 sekund stania nieruchomo,
- jedna osoba w kadrze i możliwie proste, niezatłoczone tło.

Skrypt powinien akceptować także pionowe nagrania oraz inne FPS, ale wyświetli ostrzeżenie o nietypowych parametrach zamiast odrzucać plik.

### Ustalone domyślne parametry algorytmu

Wartości będą przechowywane w konfiguracji, aby można je było zmieniać bez modyfikowania kodu:

- minimalna pewność punktu pozy: `0.5`,
- minimalna pewność detekcji osoby: `0.5`,
- wygładzanie pomiarów: mediana z ostatnich 5 klatek,
- minimalny czas utrzymania błędu: `0.2 s`,
- minimalny czas fazy ruchu: `0.15 s`,
- minimalny czas między dwoma powtórzeniami: `0.3 s`,
- minimalny kąt kolana w pełnym przysiadzie: `100°` jako wartość startowa,
- ostrzeżenie o nadmiernym pochyleniu: kąt tułowia większy niż `45°` względem pionu,
- margines tolerancji dla progów: `5°` lub `5%` długości odniesienia,
- model domyślny: lekki, gotowy model YOLO Pose; ścieżka do wag pozostanie konfigurowalna.

Progi nie będą przedstawiane jako uniwersalna norma biomechaniczna. Są wartościami startowymi do kalibracji na nagraniach testowych.

## 4. Preprocessing obrazu

Preprocessing powinien być porównany na tych samych nagraniach, zamiast dodawania wielu filtrów bez pomiaru. Możliwe kroki:

- zmiana rozmiaru klatki do ustalonej rozdzielczości,
- konwersja do RGB, jeśli wymaga tego model,
- normalizacja jasności,
- delikatne odszumianie,
- poprawa kontrastu, np. CLAHE,
- opcjonalnie rozmycie tła lub wycięcie obszaru osoby.

Należy unikać zbyt agresywnego wyostrzania i filtrów, które zmieniają kontury ciała. W raporcie warto pokazać przykładową klatkę przed preprocessingiem i po nim.

## 5. Analiza ruchu

### 5.1. Kąty

Z punktów `biodro`, `kolano`, `kostka` obliczamy kąt kolana. Z punktów `bark`, `biodro`, `kolano` obliczamy pochylenie tułowia. Można również mierzyć odległość kolana od osi stopy po normalizacji względem długości nogi.

### 5.2. Fazy przysiadu

Przykładowy automat stanów:

1. `start` — osoba stoi względnie wyprostowana,
2. `descending` — środek biodra obniża się,
3. `bottom` — osiągnięto minimalną wysokość,
4. `ascending` — biodro podnosi się,
5. `completed` — powrót do pozycji stojącej.

Do ograniczenia szumu należy użyć wygładzania sygnałów, minimalnej liczby klatek w fazie oraz histerezy progów.

### 5.3. Ocena błędów

Każdy błąd powinien mieć:

- nazwę,
- warunek wykrycia,
- próg,
- czas trwania lub liczbę kolejnych klatek,
- komunikat dla użytkownika,
- poziom pewności.

Nie należy zgłaszać błędu na podstawie pojedynczej niestabilnej predykcji punktu.

W pierwszej wersji próg głębokości będzie konfigurowalny, ale podstawowa interpretacja będzie oparta na tym, czy biodro znajduje się niżej niż kolano. Jest to kryterium przybliżone i zależne od perspektywy kamery, dlatego należy zapisywać również użyte punkty oraz ich pewność w raporcie JSON.

## 6. Wynik działania skryptu

### Podgląd wideo

Na każdej klatce można wyświetlić:

- szkielet i punkty kluczowe,
- aktualną fazę ruchu,
- numer powtórzenia,
- kąty kolan i tułowia,
- bieżące ostrzeżenie,
- jakość/pewność detekcji.

### Raport

Po zakończeniu analizy raport może zawierać:

- liczbę wykrytych powtórzeń,
- czas trwania każdego powtórzenia,
- minimalny kąt kolana,
- maksymalne pochylenie tułowia,
- liczbę i czas trwania wykrytych błędów,
- informację o klatkach, na których nie wykryto poprawnie pozy.

W tej wersji zamiast CSV zapisujemy raport JSON. Jego podstawowa struktura:

```json
{
  "input": {"file": "nazwa.mp4", "fps": 30, "width": 1920, "height": 1080},
  "summary": {"repetitions": 3, "uncertain_frames": 4},
  "repetitions": [
    {
      "index": 1,
      "start_frame": 120,
      "end_frame": 185,
      "min_knee_angle_deg": 96.4,
      "max_torso_lean_deg": 38.2,
      "errors": [
        {"type": "insufficient_depth", "start_frame": 150, "end_frame": 164}
      ]
    }
  ],
  "configuration": {"knee_angle_threshold_deg": 100}
}
```

Wartości pomiarowe i błędy powinny zawierać numer klatki, aby można było odnaleźć zdarzenie na filmie wynikowym.

## 7. Dane testowe i ocena jakości

Minimalny zestaw testowy powinien zawierać kilka nagrań:

- poprawne przysiady,
- zbyt płytkie przysiady,
- nadmierne pochylenie tułowia,
- kolana uciekające do środka,
- nagrania z gorszym oświetleniem,
- częściowo zasłoniętą sylwetkę.

Dla każdego filmu warto ręcznie oznaczyć liczbę powtórzeń i występujące błędy. Do oceny można użyć:

- dokładności liczenia powtórzeń,
- precision/recall dla wykrywania błędów,
- zgodności faz ruchu,
- liczby fałszywych ostrzeżeń,
- czasu analizy względem długości filmu.

Nie wystarczy ocenić skryptu na jednym filmie autora, ponieważ wynik może być przypadkowo dopasowany do konkretnego tła i osoby.

## 8. Proponowana struktura projektu

```text
projekt/
├── input/
├── output/
├── src/
│   ├── video.py          # odczyt i zapis filmu
│   ├── preprocessing.py  # przygotowanie klatek
│   ├── pose.py           # detekcja punktów
│   ├── geometry.py       # kąty i odległości
│   ├── squat.py          # fazy, powtórzenia i reguły
│   └── main.py           # uruchomienie pipeline'u
├── tests/
├── requirements.txt
└── README.md
```

Nazwy mogą zostać zmienione, jeśli finalna implementacja będzie prostsza.

## 9. Plan realizacji

1. Przygotować przykładowe nagrania i ustalić warunki nagrywania.
2. Zbudować odczyt filmu i zapis klatek wynikowych.
3. Dodać preprocessing oraz wizualizację.
4. Dodać model pozy/YOLO Pose i sprawdzić jakość punktów.
5. Zaimplementować obliczanie kątów.
6. Wykryć fazy ruchu i liczenie powtórzeń.
7. Dodać reguły błędów i wygładzanie.
8. Dodać raport JSON oraz podsumowanie.
9. Przetestować na oznaczonych filmach.
10. Opisać ograniczenia, wyniki i źródła w README.

### Sposób uruchamiania w GitHub Codespaces

Pierwsza wersja będzie uruchamiana z terminala:

```bash
python -m src.main --input input/przysiad.mp4 --output output/przysiad_analyzed.mp4 \
  --report output/przysiad_report.json
```

Parametry modelu i progów będą mogły być przechowywane w pliku `config.yaml`. Skrypt nie będzie wymagał przesyłania filmu do przeglądarki ani usługi zewnętrznej. Pliki wejściowe i wynikowe pozostaną w katalogach `input/` i `output/`.

## 10. Założenia do weryfikacji podczas testów

1. Czy na filmie zawsze znajduje się jedna osoba?
2. Czy całe ciało i sztanga pozostają widoczne przez cały czas?
3. Czy widok z boku zapewnia wystarczającą wiarygodność oceny kolan?
4. Czy wartości progów wymagają osobnej kalibracji dla konkretnej osoby?
5. Czy jakość gotowego modelu YOLO Pose jest wystarczająca bez własnego trenowania?

## 11. Kryterium gotowości pierwszej wersji

Pierwsza wersja jest gotowa, gdy dla ustalonego typu nagrania:

- poprawnie odczytuje film i nie przerywa analizy przy pojedynczej słabej klatce,
- wykrywa większość powtórzeń,
- pokazuje punkty i kąty na filmie wynikowym,
- rozróżnia co najmniej trzy podstawowe przypadki: poprawny przysiad, zbyt płytki przysiad i nadmierne pochylenie tułowia,
- zapisuje wynik oraz jasno komunikuje ograniczenia metody.

