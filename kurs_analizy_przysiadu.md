# Kurs: analiza przysiadu z filmu

## Od nagrania telefonem do własnego analizatora w Pythonie

### Dla kogo jest ten kurs?

Ten kurs prowadzi krok po kroku przez budowę projektu z zakresu computer vision. Celem nie jest skopiowanie gotowego programu, tylko zrozumienie, jak samodzielnie zaplanować i zaimplementować analizę przysiadu.

Po ukończeniu kursu powinieneś rozumieć:

- jak komputer reprezentuje film i pojedynczą klatkę,
- do czego służą OpenCV, YOLO i YOLO Pose,
- czym są punkty kluczowe ciała,
- jak obliczać kąty i odległości,
- jak wykrywać fazy ruchu i liczyć powtórzenia,
- jak ograniczać fałszywe ostrzeżenia,
- jak testować taki system i opisywać jego ograniczenia.

Projekt analizuje lokalny plik z telefonu. Kamera jest nieruchoma, ustawiona z boku, a na filmie znajduje się jedna osoba wykonująca przysiad ze sztangą.

> **Ważne:** wynik jest orientacyjną analizą techniki, a nie poradą medyczną ani zastępstwem trenera.

---

## 1. Co dokładnie budujemy?

Nasz system będzie miał następujący przepływ:

```text
plik wideo
    ↓
odczyt klatek przez OpenCV
    ↓
preprocessing obrazu
    ↓
YOLO Pose: punkty kluczowe ciała
    ↓
kąty, odległości i wygładzanie pomiarów
    ↓
fazy ruchu i liczenie powtórzeń
    ↓
reguły błędów
    ↓
film z komunikatami + raport JSON
```

W pierwszej wersji nie trenujemy własnej sztucznej inteligencji. Korzystamy z gotowego modelu pozy, a własną logikę budujemy na podstawie punktów ciała i reguł geometrycznych. Dzięki temu łatwiej zrozumieć, dlaczego system zgłosił konkretny problem.

### Co system ma wykrywać?

W ustalonym zakresie:

1. zbyt małą głębokość przysiadu,
2. nadmierne pochylenie tułowia,
3. kolano uciekające do środka,
4. odrywanie pięt,
5. liczbę powtórzeń i fazy ruchu.

Ocena kolana uciekającego do środka z widoku bocznego jest ograniczona. Ten problem jest znacznie łatwiejszy do oceny z przodu. Dlatego traktujemy go jako funkcję o niższej pewności i możliwe rozszerzenie o drugi widok.

---

## 2. Przygotowanie nagrania telefonem

Jakość nagrania często ma większy wpływ na wynik niż wybór filtra lub progu.

### Zalecane ustawienie

- telefon poziomo, najlepiej na statywie,
- rozdzielczość 1080p,
- 30 klatek na sekundę,
- kamera z boku, prostopadle do płaszczyzny ruchu,
- odległość około 3–4 metrów,
- całe ciało, stopy i sztanga widoczne z zapasem,
- brak zoomu cyfrowego,
- stabilne oświetlenie z przodu lub z boku,
- jedna osoba w kadrze,
- proste, niezatłoczone tło.

Przed serią nagraj około 2 sekundy spokojnego stania, a po serii kolejne 2 sekundy. Ułatwi to wykrycie pozycji początkowej i końcowej.

### Dlaczego widok z boku?

Z boku łatwo obserwować:

- kąt kolana,
- pochylenie tułowia,
- położenie biodra względem kolana,
- kontakt pięty z podłożem.

Z boku trudniej wiarygodnie ocenić ruch kolana do środka. Nie należy ukrywać tego ograniczenia w raporcie.

### Zadanie 1

Nagraj krótki film z 3–5 powtórzeniami. Sprawdź ręcznie:

- czy nie ucięto stóp,
- czy sztanga nie znika poza kadrem,
- czy osoba nie jest zasłonięta,
- czy oświetlenie nie tworzy silnego cienia,
- czy telefon nie porusza się podczas serii.

---

## 3. Środowisko i biblioteki

Projekt może działać w GitHub Codespaces. Podstawowe narzędzia:

- **Python** — język projektu,
- **OpenCV** — odczyt, zapis i rysowanie na klatkach,
- **Ultralytics** — uruchamianie modelu YOLO Pose,
- **NumPy** — obliczenia wektorowe i tablice,
- **JSON** — zapis raportu.

Przykładowe przygotowanie środowiska:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python ultralytics numpy
```

W Codespaces warto zapisać zależności w `requirements.txt`, aby projekt dało się odtworzyć:

```text
opencv-python
ultralytics
numpy
```

Przed użyciem konkretnego modelu sprawdź jego dokumentację, nazwę pliku wag i licencję. Wagi modelu nie powinny być przypadkowo dołączane do repozytorium, jeśli są duże lub mają ograniczenia licencyjne.

---

## 4. Film jako zbiór klatek

Film nie jest dla programu jednym „obrazkiem”. Jest sekwencją klatek:

```text
klatka 0 → klatka 1 → klatka 2 → ... → klatka N
```

OpenCV pozwala odczytać metadane filmu:

- szerokość,
- wysokość,
- FPS,
- liczbę klatek.

FPS jest ważny, bo próg „błąd trwa 0,2 sekundy” trzeba przeliczyć na klatki:

```text
liczba_klatek = czas_w_sekundach × FPS
```

Dla filmu 30 FPS wartość 0,2 sekundy oznacza około 6 kolejnych klatek.

### Pseudokod odczytu

```text
otwórz film
odczytaj FPS i rozmiar
otwórz plik wynikowy

dopóki są klatki:
    pobierz klatkę
    przetwórz klatkę
    zapisz klatkę wynikową

zamknij film i plik wynikowy
```

Najpierw warto uruchomić sam odczyt i zapis filmu, bez modelu. To rozdziela problemy z plikiem od problemów z detekcją.

---

## 5. Preprocessing obrazu

Preprocessing to przygotowanie obrazu przed przekazaniem go do modelu. Nie jest magicznym sposobem na naprawienie każdego nagrania.

### Rozsądne kroki

1. zmiana rozmiaru do rozdzielczości używanej przez model,
2. konwersja BGR → RGB, jeśli wymaga tego biblioteka,
3. delikatna normalizacja jasności,
4. opcjonalna poprawa kontrastu, np. CLAHE,
5. zachowanie proporcji obrazu.

Nie stosuj agresywnego wyostrzania ani filtrów zmieniających kontury ciała. Zawsze porównuj wynik przed i po preprocessingu na tym samym filmie.

### Zadanie 2

Zapisz kilka przykładowych klatek:

- oryginalną,
- po zmianie rozmiaru,
- po poprawie kontrastu.

Porównaj, czy punkty ciała są faktycznie łatwiejsze do wykrycia. Jeśli nie, usuń niepotrzebny filtr.

---

## 6. YOLO Pose i punkty kluczowe

Zwykła detekcja obiektu odpowiada na pytanie: „gdzie znajduje się osoba?”. Model pose-estimation idzie dalej i próbuje wskazać punkty ciała.

Potrzebne punkty to między innymi:

- bark,
- biodro,
- kolano,
- kostka,
- pięta,
- palce stopy.

Każdy punkt ma zwykle:

- współrzędną `x`,
- współrzędną `y`,
- pewność detekcji.

Przykładowo:

```text
biodro = (x=640, y=420, confidence=0.93)
```

Nie używaj punktu bez sprawdzenia jego pewności. Jeśli model nie widzi stopy, nie udawaj, że punkt jest poprawny.

### Zasada niepewnej detekcji

W naszej wersji:

- punkty poniżej progu pewności są uznawane za niepewne,
- taka klatka jest pomijana w obliczeniach,
- niepewna klatka nie jest automatycznie błędem techniki,
- liczba niepewnych klatek trafia do raportu.

Domyślny próg startowy: `0.5`. Traktuj go jako parametr do testów, nie jako prawdę uniwersalną.

---

## 7. Geometria: kąty i odległości

### Kąt w punkcie

Kąt kolana wyznaczamy z trzech punktów:

```text
biodro — kolano — kostka
```

Kolano jest wierzchołkiem kąta. Kąt bliski 180° oznacza nogę prawie wyprostowaną. Mniejszy kąt oznacza głębsze ugięcie.

Kąt tułowia można przybliżyć na podstawie:

```text
bark — biodro
```

i porównania tego odcinka z kierunkiem pionowym.

### Dlaczego nie wystarczy odległość w pikselach?

Odległość w pikselach zależy od:

- odległości osoby od telefonu,
- zoomu,
- rozdzielczości,
- perspektywy.

Kąty są zwykle bardziej odporne na zmianę skali. Dla odległości warto używać normalizacji, np. dzielić odległość kolana od osi stopy przez długość nogi.

### Głębokość

W pierwszej wersji głębokość opiera się na relacji biodra względem kolana. Ponieważ obraz jest dwuwymiarowy, wynik zależy od ustawienia kamery i budowy osoby. Próg powinien być konfigurowalny i sprawdzony na filmach testowych.

---

## 8. Fazy ruchu i liczenie powtórzeń

Nie analizujemy każdej klatki jako niezależnego zdjęcia. Musimy rozumieć kolejność ruchu.

### Proponowane fazy

1. `start` — osoba stoi względnie wyprostowana,
2. `descending` — biodro obniża się,
3. `bottom` — osiągnięto najniższą pozycję,
4. `ascending` — biodro podnosi się,
5. `completed` — osoba wróciła do pozycji stojącej.

### Automat stanów

Najprostszy automat przechodzi do następnego stanu tylko wtedy, gdy warunek utrzyma się odpowiednio długo. Dzięki temu pojedynczy skok punktu nie tworzy fałszywego powtórzenia.

Przykład:

```text
start → descending → bottom → ascending → completed
```

Powtórzenie zaliczamy dopiero po przejściu przez `bottom` i powrocie do `completed`.

### Wygładzanie

Pomiary z modelu mogą drżeć. Można użyć mediany z ostatnich 5 wartości:

```text
[98, 96, 130, 97, 95] → mediana = 97
```

Mediana jest odporna na pojedynczy skok. Nie należy jednak wygładzać tak mocno, aby opóźniać wykrycie fazy.

---

## 9. Reguły wykrywania błędów

Każda reguła powinna mieć:

- nazwę,
- wymagane punkty,
- warunek,
- próg,
- minimalny czas trwania,
- poziom pewności,
- komunikat dla użytkownika.

### Zbyt płytki przysiad

Jeżeli biodro nie osiągnie ustalonej relacji względem kolana w fazie `bottom`, zapisujemy `insufficient_depth`.

### Nadmierne pochylenie tułowia

Jeżeli kąt tułowia przekroczy wartość startową 45° względem pionu i utrzyma się przez kilka klatek, zapisujemy `excessive_torso_lean`.

### Odrywanie pięt

Odrywanie pięt jest trudne, jeśli punkt pięty jest niestabilny lub stopa jest zasłonięta. Reguła powinna działać tylko przy wystarczającej pewności punktów.

### Kolano uciekające do środka

Z widoku bocznego ta reguła ma ograniczoną wiarygodność. Nie przedstawiaj jej jako pewnej diagnozy. W raporcie warto zapisać niższy poziom pewności albo osobno oznaczyć ją jako eksperymentalną.

### Histereza i czas trwania

Nie zgłaszaj błędu na podstawie jednej klatki. Domyślna wartość to 0,2 sekundy. Dla 30 FPS daje to około 6 kolejnych klatek.

---

## 10. Film wynikowy i raport JSON

Na filmie wynikowym warto wyświetlać:

- szkielet,
- numer powtórzenia,
- aktualną fazę,
- kąty kolana i tułowia,
- komunikat błędu,
- informację o niepewnej detekcji.

Nie koloruj całej klatki na czerwono bez wyjaśnienia. Komunikat powinien mówić, jaki warunek został naruszony.

### Przykładowy raport

```json
{
  "input": {
    "file": "przysiad.mp4",
    "fps": 30,
    "width": 1920,
    "height": 1080
  },
  "summary": {
    "repetitions": 3,
    "uncertain_frames": 4
  },
  "repetitions": [
    {
      "index": 1,
      "start_frame": 120,
      "end_frame": 185,
      "min_knee_angle_deg": 96.4,
      "max_torso_lean_deg": 38.2,
      "errors": [
        {
          "type": "insufficient_depth",
          "start_frame": 150,
          "end_frame": 164
        }
      ]
    }
  ],
  "configuration": {
    "knee_angle_threshold_deg": 100
  }
}
```

Numer klatki jest ważny, bo pozwala odnaleźć zdarzenie na filmie.

---

## 11. Testowanie i kalibracja

Nie wystarczy sprawdzić programu na jednym własnym filmie.

### Zestaw testowy

Przygotuj osobne nagrania z:

- poprawnymi przysiadami,
- zbyt płytkimi przysiadami,
- nadmiernym pochyleniem,
- odrywaniem pięt,
- podejrzanym ruchem kolan,
- gorszym oświetleniem,
- częściowym zasłonięciem sylwetki.

Dla każdego filmu ręcznie zapisz:

- liczbę powtórzeń,
- momenty rozpoczęcia i zakończenia,
- występujące problemy,
- sytuacje, w których człowiek nie jest pewien oceny.

### Co mierzyć?

- błąd liczby powtórzeń,
- liczbę fałszywych ostrzeżeń,
- liczbę pominiętych błędów,
- zgodność faz ruchu,
- liczbę niepewnych klatek,
- czas analizy względem długości filmu.

Nie dostrajaj progów na tym samym filmie, na którym później raportujesz wynik. Rozdziel filmy kalibracyjne i testowe.

---

## 12. Proponowana struktura projektu

```text
projekt/
├── input/
├── output/
├── src/
│   ├── video.py
│   ├── preprocessing.py
│   ├── pose.py
│   ├── geometry.py
│   ├── squat.py
│   └── main.py
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

Odpowiedzialności:

- `video.py` — otwieranie, czytanie i zapis filmu,
- `preprocessing.py` — przygotowanie klatek,
- `pose.py` — uruchomienie YOLO Pose i walidacja punktów,
- `geometry.py` — kąty, wektory i odległości,
- `squat.py` — fazy, powtórzenia i reguły błędów,
- `main.py` — połączenie wszystkich elementów.

Podział jest ważny: jeśli wszystko znajdzie się w jednym pliku, trudniej będzie testować i poprawiać pojedyncze elementy.

---

## 13. Plan samodzielnej pracy

Realizuj projekt w tej kolejności:

1. przygotuj środowisko i odtwórz film,
2. zapisz kopię filmu bez zmian,
3. dodaj rysowanie prostego tekstu na klatce,
4. dodaj preprocessing,
5. uruchom YOLO Pose i narysuj punkty,
6. zapisz punkty oraz ich pewność,
7. oblicz kąt kolana na pojedynczej klatce,
8. oblicz kąt na całym filmie i narysuj wykres lub wartości,
9. dodaj wygładzanie,
10. dodaj automat faz,
11. dodaj liczenie powtórzeń,
12. dodaj jedną regułę błędu,
13. dodaj kolejne reguły,
14. dodaj JSON,
15. przetestuj na osobnych filmach.

Nie dodawaj wszystkich funkcji naraz. Po każdym kroku uruchom program i sprawdź wynik wizualnie.

---

## 14. Typowe problemy i sposób myślenia

### Model nie widzi punktów

Sprawdź kolejno:

1. czy osoba jest wystarczająco duża w kadrze,
2. czy oświetlenie jest poprawne,
3. czy punkty nie są zasłonięte przez sztangę,
4. czy nie pomyliłeś kolejności kanałów BGR/RGB,
5. czy próg pewności nie jest zbyt wysoki.

### Liczba powtórzeń jest zła

Sprawdź:

- czy fazy nie przełączają się od pojedynczych skoków,
- czy używasz wygładzania,
- czy masz minimalny czas fazy,
- czy powtórzenie jest liczone dopiero po pełnym cyklu.

### Zbyt dużo fałszywych błędów

Nie obniżaj od razu wszystkich progów. Najpierw sprawdź:

- czy punkt ma odpowiednią pewność,
- czy błąd trwa wystarczająco długo,
- czy kamera jest ustawiona zgodnie z wymaganiami,
- czy próg odpowiada temu, co naprawdę chcesz wykrywać.

### Wynik działa tylko dla autora

To oznacza, że testy były zbyt wąskie. Dodaj inne osoby, ubrania, oświetlenie, tempo ruchu i niewielkie różnice ustawienia kamery.

---

## 15. Ograniczenia projektu

Należy jasno opisać, że:

- system analizuje obraz 2D, a ciało porusza się w 3D,
- widok z boku ogranicza ocenę kolan,
- model może mylić punkty przy zasłonięciu,
- progi nie są uniwersalną normą dla każdej osoby,
- sztanga może zasłaniać barki lub ręce,
- wynik nie jest diagnozą medyczną,
- jakość nagrania wpływa na wiarygodność rezultatu.

Dobre omówienie ograniczeń jest częścią projektu, a nie przyznaniem się do porażki.

---

## 16. Kryterium ukończenia pierwszej wersji

Pierwsza wersja jest gotowa, gdy:

- przyjmuje plik wideo z telefonu,
- nie przerywa działania przez pojedynczą niepewną klatkę,
- rysuje punkty i szkielet,
- rozpoznaje podstawowe fazy,
- liczy większość powtórzeń w filmach testowych,
- wykrywa zbyt małą głębokość i nadmierne pochylenie,
- zapisuje film wynikowy,
- zapisuje poprawny JSON,
- komunikuje ograniczenia i niepewne detekcje.

Przykładowe uruchomienie końcowe:

```bash
python -m src.main \
  --input input/przysiad.mp4 \
  --output output/przysiad_analyzed.mp4 \
  --report output/przysiad_report.json
```

Najważniejsza zasada kursu: najpierw spraw, aby system poprawnie mierzył i pokazywał dane, a dopiero potem każ mu wydawać oceny. Dzięki temu łatwiej znaleźć błąd i wyjaśnić działanie projektu.

---

## 17. Warsztat praktyczny: kod pisany etapami

Ten rozdział zawiera realny kod. Nie kopiuj wszystkiego do jednego pliku. Utwórz pliki w podanej kolejności i uruchamiaj test po każdym etapie.

### Etap 0: przygotowanie projektu

Utwórz strukturę:

```text
projekt/
├── input/
├── output/
├── src/
│   ├── __init__.py
│   └── step01_video.py
└── requirements.txt
```

Plik `requirements.txt`:

```text
opencv-python
ultralytics
numpy
```

Instalacja:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Na Windows aktywacja środowiska to:

```powershell
.venv\Scripts\activate
```

### Etap 1: odczyt i zapis filmu

Najpierw sprawdź, czy OpenCV potrafi odczytać film i zapisać jego kopię. Na tym etapie nie używamy jeszcze AI.

Plik `src/step01_video.py`:

```python
from pathlib import Path

import cv2


INPUT = Path("input/przysiad.mp4")
OUTPUT = Path("output/kopia.mp4")


def copy_video(input_path: Path, output_path: Path) -> None:
    """Czyta film klatka po klatce i zapisuje jego kopię."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nie można otworzyć filmu: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Film ma niepoprawne metadane.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Nie można utworzyć pliku: {output_path}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        # Zasoby muszą zostać zamknięte także po błędzie.
        capture.release()
        writer.release()


if __name__ == "__main__":
    copy_video(INPUT, OUTPUT)
    print(f"Zapisano: {OUTPUT}")
```

Uruchom:

```bash
python -m src.step01_video
```

**Sprawdź:** czy `output/kopia.mp4` odtwarza się i ma tę samą długość co oryginał. Jeśli nie, nie przechodź dalej — najpierw napraw obsługę pliku.

### Etap 2: rysowanie informacji na klatce

Dodaj prosty napis, aby upewnić się, że potrafisz tworzyć film wynikowy. Skopiuj etap 1 do `src/step02_overlay.py` i zmień pętlę odczytu:

```python
while True:
    ok, frame = capture.read()
    if not ok:
        break

    cv2.putText(
        frame,
        "Analiza przysiadu",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    writer.write(frame)
```

Nie dodawaj jeszcze reguł techniki. Ten etap uczy podstawowej pętli: pobierz → zmień → zapisz.

### Etap 3: pierwszy model YOLO Pose

Dodaj plik `src/step03_pose.py`. Model zostanie pobrany przy pierwszym uruchomieniu, więc potrzebujesz połączenia z internetem tylko podczas pobierania wag.

```python
from pathlib import Path

import cv2
from ultralytics import YOLO


INPUT = Path("input/przysiad.mp4")
OUTPUT = Path("output/pose.mp4")


def draw_pose(input_path: Path, output_path: Path) -> None:
    # Lekki model jest dobry do pierwszego prototypu.
    model = YOLO("yolo11n-pose.pt")
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"Nie można otworzyć filmu: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # verbose=False wyłącza log dla każdej klatki.
            result = model.predict(frame, conf=0.5, verbose=False)[0]
            annotated = result.plot()
            writer.write(annotated)
    finally:
        capture.release()
        writer.release()


if __name__ == "__main__":
    draw_pose(INPUT, OUTPUT)
```

Uruchom:

```bash
python -m src.step03_pose
```

**Sprawdź:** czy na filmie widać punkty i szkielet. Nie przechodź do kątów, dopóki model nie działa na Twoim nagraniu.

> Jeśli nazwa modelu zmieni się w dokumentacji Ultralytics, zapisz ścieżkę do wag w konfiguracji zamiast wpisywać ją w wielu plikach. Zawsze sprawdź licencję modelu.

### Etap 4: odczyt punktów i pewności

Samo `result.plot()` rysuje obraz, ale do obliczeń potrzebujemy danych. W YOLO Pose punkty są dostępne przez `result.keypoints`.

Dodaj funkcję do `src/pose.py`:

```python
from typing import Optional

import numpy as np


def get_first_person_points(
    result,
    confidence_threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """Zwraca punkty pierwszej osoby albo None przy niepewnej detekcji."""
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    points = result.keypoints.xy[0].cpu().numpy()
    confidence = result.keypoints.conf[0].cpu().numpy()

    # Cała klatka jest pomijana, gdy brakuje kluczowych punktów.
    required = [5, 11, 12, 13, 14, 15, 16]
    if any(confidence[index] < confidence_threshold for index in required):
        return None

    return points
```

Indeksy punktów zależą od używanego formatu pozy. Przed dalszą pracą sprawdź mapę indeksów w dokumentacji modelu. Nie zakładaj, że każdy model używa identycznej numeracji.

### Etap 5: obliczanie kąta

Utwórz `src/geometry.py`:

```python
import math

import numpy as np


def angle_at_vertex(
    first: np.ndarray,
    vertex: np.ndarray,
    last: np.ndarray,
) -> float:
    """Zwraca kąt w punkcie vertex w stopniach."""
    vector_a = first - vertex
    vector_b = last - vertex
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        raise ValueError("Nie można obliczyć kąta dla zbieżnych punktów.")

    cosine = np.dot(vector_a, vector_b) / denominator
    # Ograniczenie chroni przed błędem numerycznym, np. 1.0000001.
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def knee_angle(points: np.ndarray, side: str = "left") -> float:
    """Oblicza kąt kolana na podstawie punktów modelu COCO."""
    if side == "left":
        hip, knee, ankle = 11, 13, 15
    elif side == "right":
        hip, knee, ankle = 12, 14, 16
    else:
        raise ValueError("side musi mieć wartość 'left' albo 'right'.")

    return angle_at_vertex(points[hip], points[knee], points[ankle])
```

Przetestuj tę funkcję bez filmu:

```python
import numpy as np

from src.geometry import angle_at_vertex


angle = angle_at_vertex(
    np.array([0.0, 1.0]),
    np.array([0.0, 0.0]),
    np.array([1.0, 0.0]),
)
print(angle)  # Oczekiwane około 90 stopni.
```

Najpierw testuj matematykę na prostych punktach. Dopiero potem łącz ją z modelem.

### Etap 6: wygładzanie pomiarów

Punkty modelu mogą drżeć. Utwórz `src/smoothing.py`:

```python
from collections import deque

import numpy as np


class MedianSmoother:
    """Wygładza pojedynczy pomiar medianą z ostatnich wartości."""

    def __init__(self, size: int = 5):
        if size < 1:
            raise ValueError("Rozmiar wygładzania musi być dodatni.")
        self.values = deque(maxlen=size)

    def update(self, value: float) -> float:
        self.values.append(value)
        return float(np.median(np.array(self.values)))
```

Użycie:

```python
from src.smoothing import MedianSmoother

smoother = MedianSmoother(size=5)
for raw_angle in [98, 97, 150, 96, 95]:
    print(smoother.update(raw_angle))
```

Wartość 150 jest pojedynczym skokiem i nie powinna całkowicie zmienić wyniku.

### Etap 7: najprostszy automat faz

Utwórz `src/squat.py`:

```python
from enum import Enum


class Phase(str, Enum):
    START = "start"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"
    COMPLETED = "completed"


class SquatState:
    def __init__(self, deep_angle: float = 100.0):
        self.phase = Phase.START
        self.deep_angle = deep_angle
        self.repetitions = 0

    def update(self, knee_angle: float, previous_angle: float | None) -> Phase:
        if previous_angle is None:
            return self.phase

        going_down = knee_angle < previous_angle
        going_up = knee_angle > previous_angle

        if self.phase == Phase.START and going_down:
            self.phase = Phase.DESCENDING
        elif self.phase == Phase.DESCENDING and knee_angle <= self.deep_angle:
            self.phase = Phase.BOTTOM
        elif self.phase == Phase.BOTTOM and going_up:
            self.phase = Phase.ASCENDING
        elif self.phase == Phase.ASCENDING and knee_angle > 160:
            self.phase = Phase.COMPLETED
            self.repetitions += 1
            self.phase = Phase.START

        return self.phase
```

To celowo mały automat. Następnie dodaj minimalny czas fazy, histerezę i obsługę niepewnych klatek. Nie próbuj rozwiązać wszystkich przypadków w pierwszej wersji.

### Etap 8: jedna reguła błędu

Najpierw zaimplementuj tylko głębokość. Później dodawaj kolejne reguły.

```python
def insufficient_depth(knee_angle: float, threshold: float = 100.0) -> bool:
    """Zwraca True, gdy ugięcie jest mniejsze od ustalonego progu."""
    return knee_angle > threshold
```

W praktycznym automacie tę funkcję wywołuj dopiero w fazie `BOTTOM`. Jeśli wywołasz ją w każdej klatce, zgłosisz błąd również podczas stania.

### Etap 9: zapis raportu JSON

Utwórz `src/report.py`:

```python
import json
from pathlib import Path
from typing import Any


def save_report(report: dict[str, Any], path: Path) -> None:
    """Zapisuje raport w formacie czytelnym dla człowieka."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

Przykładowe dane:

```python
from pathlib import Path

from src.report import save_report


save_report(
    {
        "summary": {"repetitions": 1, "uncertain_frames": 0},
        "repetitions": [],
    },
    Path("output/report.json"),
)
```

### Etap 10: połączenie etapów

Dopiero teraz połącz moduły w `src/main.py`. Program powinien:

1. otworzyć film,
2. uruchomić model,
3. pobrać punkty pierwszej osoby,
4. pominąć niepewną klatkę,
5. obliczyć i wygładzić kąt,
6. zaktualizować fazę,
7. zapisać komunikat na klatce,
8. zapisać film i raport.

Nie kopiuj dużego, gotowego `main.py` z internetu. Napisz go sam, korzystając z funkcji wypróbowanych w poprzednich etapach. Po każdej dodanej funkcji uruchom krótki film testowy.

Przykładowy interfejs uruchomienia:

```bash
python -m src.main \
  --input input/przysiad.mp4 \
  --output output/przysiad_analyzed.mp4 \
  --report output/przysiad_report.json
```

### Kontrola po każdym etapie

Po każdym kroku odpowiedz sobie:

- jaki plik powstał,
- co mogę zobaczyć na obrazie,
- jakie dane mogę wydrukować,
- czy błąd kończy się jasnym komunikatem,
- czy następny etap może użyć tego wyniku.

Jeśli odpowiedź brzmi „nie wiem”, nie przechodź dalej. W tym projekcie debugowanie ma polegać na oglądaniu klatek, drukowaniu punktów i sprawdzaniu małych funkcji, a nie na zgadywaniu.
