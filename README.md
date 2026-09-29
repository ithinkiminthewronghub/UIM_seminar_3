# Cvičení 3: Nehierarchické shlukování

Toto cvičení je třetím praktickým cvičením předmětu **Umělá inteligence v medicíně**. Cílem je implementovat dva algoritmy nehierarchického (partičního) shlukování — **k-means** a **fuzzy c-means** — a aplikovat je na reálný medicínský problém: **segmentaci mikroskopického snímku buněk**. Každý pixel snímku je popsán třemi čísly (R, G, B) a algoritmem přiřazen do jednoho ze $k$ shluků — skupin pixelů se podobnou barvou, které odpovídají různým strukturám v preparátu.

Cvičení staví na kódu z Cvičení 01 (metriky vzdálenosti) a prohloubí znalosti z Cvičení 02 (hierarchické shlukování). Zatímco hierarchické shlukování bylo deterministické a nevyžadovalo počet shluků předem, k-means a FCM jsou iterativní, závislé na inicializaci a vyžadují $k$ jako vstupní parametr.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Implementuje k-means od základů** — pochopí iterativní smyčku přiřazení a přepočtu těžišť a rozumí podmínce konvergence.
2. **Implementuje fuzzy c-means** — rozumí rozdílu mezi tvrdým (hard) a měkkým (fuzzy) přiřazením a ví, jak parametr fuzifikace $q$ ovlivňuje překryv shluků.
3. **Implementuje a porovná tři inicializační strategie** — Random Uniform, Forgy, k-means++ — a pochopí, proč inicializace zásadně ovlivňuje výsledek.
4. **Implementuje silhouetovou analýzu** — naučí se objektivně měřit kvalitu shlukování bez znalosti skutečných popisků a použít ji pro výběr optimálního $k$.
5. **Implementuje návrhový vzor šablonová metoda (Template Method)** — bázová třída `IterativeClustering` obsahuje sdílenou iterační smyčku; podtřídy přepisují pouze dva variační body.
6. **Aplikuje shlukování na segmentaci obrazu** — zpracuje reálný mikroskopický snímek buněk a vizualizuje výsledky segmentace.
7. **Pracuje s typovanou konfigurací** — čte parametry z YAML souboru přes dataclassy a přistupuje k nim přes atributy místo slovníkových klíčů.

---

## Struktura repozitáře

```
cviceni-03-template/
├── cviceni_03.py               # Hlavní pipeline — spusťte pro průběžné ověření
├── config.yaml                 # Konfigurace experimentu (YAML)
├── src/
│   ├── __init__.py             # Re-exporty balíčku (nepoupravujte)
│   ├── distance.py             # Distance (ABC), EuclideanDistance, Manhattan, Cosine
│   ├── initialization.py       # Initializer (ABC), RandomUniformInit, ForgyInit, KMeansPlusPlusInit
│   ├── base.py                 # IterativeClustering (ABC) — sdílená iterační smyčka
│   ├── kmeans.py               # KMeans — tvrdé přiřazení
│   ├── fuzzy_cmeans.py         # FuzzyCMeans — měkké přiřazení
│   └── silhouette.py           # silhouette_samples, silhouette_score
├── dataio/
│   ├── __init__.py             # Re-exporty balíčku (nepoupravujte)
│   ├── loader.py               # load_image() — předimplementováno
│   ├── features.py             # to_hsv(), select_channels()
│   ├── config_manager.py       # Dataclassy + load_config, make_initializer, validate_config
│   └── plotting.py             # Vizualizace — předimplementováno
├── data/
│   └── Bunky.png               # Mikroskopický snímek buněk (vstupní data)
├── graphs/                     # Výstupní složka pro grafy (generuje se automaticky)
├── test_cviceni_03.py          # Automatické testy (pytest)
└── requirements.txt            # Python závislosti
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`, `dataio/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje veřejné API. Díky tomu lze psát `from src import KMeans` místo `from src.kmeans import KMeans`. **Tyto soubory neupravujte.**

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Spuštění

```bash
python cviceni_03.py
```

Dokud nejsou implementovány všechny metody, pipeline vypíše `[NEDOKONCENO] název_metody` a přeskočí příslušné kroky. Toto chování je záměrné — pipeline lze spouštět průběžně i s částečnou implementací. Jedinou výjimkou je načtení konfigurace: protože `load_config` volá `validate_config`, je potřeba tuto validaci dokončit jako první — jinak se pipeline korektně ukončí hned na začátku. Jednotlivé algoritmy lze mezitím ověřovat přes `pytest`.

---

## Teoretický základ

### 1. Partíční shlukování — přehled

Nehierarchické (partíční) shlukování rozdělí $n$ bodů do přesně $k$ skupin (shluků). Na rozdíl od hierarchického shlukování:

- Počet shluků $k$ musí být zadán předem.
- Algoritmus iteruje, dokud se shluky nestabilizují — výsledek závisí na počátečním rozložení těžišť (**inicializace**).
- Opakované spuštění se stejnými daty může dát různé výsledky — **nedeterminismus je vlastností algoritmu, nikoli chybou**.

Oba algoritmy implementované v tomto cvičení sdílejí základní strukturu:

```
Inicializuj těžiště (k počátečních poloh)
Opakuj až max_iter krát:
    1. Přiřaď každý bod k těžišti (jak závisí na algoritmu)
    2. Přepočítej těžiště z nového přiřazení
    3. Pokud se těžiště téměř nepohnula → zastav (konvergence)
```

---

### 2. Algoritmus k-means

K-means minimalizuje **součet čtvercových vzdáleností** každého bodu od jeho těžiště (WCSS — Within-Cluster Sum of Squares):

$$\text{WCSS} = \sum_{c=0}^{k-1} \sum_{i \in C_c} \|\mathbf{x}_i - \boldsymbol{\mu}_c\|^2$$

kde $C_c$ je množina bodů přiřazených do shluku $c$ a $\boldsymbol{\mu}_c$ je těžiště shluku $c$.

**Krok přiřazení (tvrdé):** Každý bod se přiřadí k nejbližšímu těžišti:

$$\text{label}(i) = \arg\min_{c \in \{0,\ldots,k-1\}} d(\mathbf{x}_i, \boldsymbol{\mu}_c)$$

Výsledkem je pole celých čísel tvaru $(n,)$ s hodnotami $0 \ldots k-1$.

**Krok přepočtu těžišť:** Těžiště každého shluku se přepočítá jako aritmetický průměr přiřazených bodů:

$$\boldsymbol{\mu}_c = \frac{1}{|C_c|} \sum_{i \in C_c} \mathbf{x}_i$$

**Konvergence:** Algoritmus se zastaví, pokud je norma posunu všech těžišť pod prahem $\varepsilon$ (zde $10^{-6}$):

$$\|\boldsymbol{\mu}_c^{(t)} - \boldsymbol{\mu}_c^{(t-1)}\| < \varepsilon \quad \forall c$$

---

### 3. Algoritmus fuzzy c-means (FCM)

FCM zobecňuje k-means tím, že každý bod může **patřit do více shluků zároveň** s různou mírou příslušnosti. Výsledkem přiřazení není číslo shluku, ale **matice členství** $U$ tvaru $(n, k)$, kde $U_{ic} \in [0, 1]$ udává míru příslušnosti bodu $i$ do shluku $c$. Platí $\sum_{c=0}^{k-1} U_{ic} = 1$ pro každý bod $i$.

**Parametr fuzifikace $q$** (v kódu označen `q`, v literatuře také $m$) řídí míru překryvu shluků:
- $q \to 1$: FCM se přibližuje k-means (tvrdé hranice)
- $q = 2$: standardní FCM
- $q > 2$: velmi měkké hranice, shluky se silně překrývají

> Pozor: $q$ musí být větší než 1. Hodnota $q = 1$ způsobuje dělení nulou.

**Krok přiřazení (měkké — matice členství):**

$$U_{ic} = \frac{1}{\displaystyle\sum_{j=1}^{k} \left(\frac{d_{ic}}{d_{ij}}\right)^{\frac{2}{q-1}}}$$

kde $d_{ic} = d(\mathbf{x}_i, \boldsymbol{\mu}_c)$ je vzdálenost bodu $i$ od těžiště $c$.

Speciální případ: Pokud bod leží přesně na těžišti ($d_{ic} = 0$), dostane plné členství v tomto shluku ($U_{ic} = 1$, ostatní $= 0$).

**Krok přepočtu těžišť (vážený průměr):**

$$\boldsymbol{\mu}_c = \frac{\displaystyle\sum_{i=1}^{n} U_{ic}^q \cdot \mathbf{x}_i}{\displaystyle\sum_{i=1}^{n} U_{ic}^q}$$

**Konvergence:** Stejné kritérium jako u k-means — sleduje posun těžišť, nikoli změnu matice $U$. Toto jednotné kritérium umožňuje sdílenou implementaci v `IterativeClustering`.

**Tvrdé popisky z FCM:** Pro vizualizaci a silhouetovou analýzu se měkká matice $U$ převede na tvrdé popisky výběrem shluku s nejvyšším členstvím:

$$\text{label}(i) = \arg\max_{c} U_{ic}$$

---

### 4. Inicializační strategie

Výsledek k-means i FCM závisí na počátečním rozmístění těžišť. Tři implementované strategie se liší kvalitou a výpočetními nároky:

#### Random Uniform

Těžiště jsou generována **náhodně z rovnoměrného rozdělení** v rozsahu dat. Pro každý příznak $j$:

$$\mu_{cj} \sim \mathcal{U}(\min_i x_{ij},\ \max_i x_{ij})$$

Těžiště neodpovídají žádnému skutečnému bodu dat. Jednoduchá, rychlá, ale náchylná k špatné inicializaci.

#### Forgy

Těžiště jsou $k$ **náhodně vybraných existujících bodů** z datasetu (bez opakování). Bod je vždy v prostoru dat — lepší start než Random Uniform, ale stále nedeterministická.

#### K-means++

Chytré rozmístění těžišť snižuje pravděpodobnost uváznutí v lokálním minimu:

1. Vyberte první těžiště náhodně (rovnoměrně) z bodů $x$.
2. Pro každý bod $i$ vypočítejte kvadratickou vzdálenost k nejbližšímu dosud zvolenému těžišti: $D(i)^2$.
3. Vyberte další těžiště s pravděpodobností úměrnou $D(i)^2$ — vzdálenější body mají větší šanci být vybráni.
4. Opakujte kroky 2–3, dokud nemáte $k$ těžišť.

K-means++ je výchozí inicializace scikit-learn implementace. Má vyšší výpočetní cenu při inicializaci, ale statisticky dosahuje lepšího výsledku po konvergenci.

---

### 5. Silhouetová analýza

Silhouetová hodnota $s(i)$ měří, jak dobře bod $i$ patří do svého shluku **v porovnání s nejbližším jiným shlukem**, aniž by bylo třeba znát skutečné popisky:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

kde:

- $a(i)$ = průměrná vzdálenost bodu $i$ od všech ostatních bodů ve **stejném** shluku (míra soudržnosti — nižší je lepší),
- $b(i)$ = průměrná vzdálenost bodu $i$ od všech bodů v **nejbližším jiném** shluku (míra oddělenosti — vyšší je lepší).

| Hodnota $s(i)$ | Interpretace |
|:---:|:---|
| blízká $+1$ | Bod je dobře umístěn ve svém shluku |
| blízká $0$ | Bod leží na hranici dvou shluků |
| blízká $-1$ | Bod pravděpodobně patří do jiného shluku |

Speciální případ: Pokud shluk obsahuje jediný bod, $s(i) = 0$ (silhoueta není definována).

**Průměrné silhouetové skóre** $\bar{s}$ je průměr $s(i)$ přes všechny body — jednočíselná míra kvality shlukování. Porovnáním $\bar{s}$ pro různá $k$ lze identifikovat optimální počet shluků (maximum křivky).

---

### 6. Vzor šablonová metoda (Template Method)

K-means a FCM sdílejí **identickou iterační smyčku** — liší se pouze ve dvou krocích:

| Krok | K-means | FCM |
|:---|:---|:---|
| Přiřazení | `argmin` vzdáleností → tvrdé popisky `(n,)` | vzorec FCM → matice členství `(n, k)` |
| Přepočet těžišť | aritmetický průměr skupiny | vážený průměr s vahami $U^q$ |

Návrhový vzor **Template Method** tuto situaci řeší elegantně: `IterativeClustering` (abstraktní třída) implementuje celou smyčku a deklaruje `_update_assignment` a `_update_centroids` jako abstraktní metody. Každá podtřída (`KMeans`, `FuzzyCMeans`) vyplní pouze tyto dvě metody — smyčku nepíše znovu.

```
IterativeClustering (ABC)           # sdílí: fit(), _distances_to_centroids(), _has_converged()
    │
    ├── KMeans                      # implementuje: _update_assignment(), _update_centroids(), predict()
    └── FuzzyCMeans                 # implementuje: _update_assignment(), _update_centroids(), predict()
```

---

### 7. Metriky vzdálenosti

Stejné třídy jako v Cvičení 01, zkopírované do `src/distance.py`:

| Třída | Vzorec | Je metrikou? |
|:---|:---|:---:|
| `EuclideanDistance` | $\sqrt{\sum_j (a_j - b_j)^2}$ | Ano |
| `ManhattanDistance` | $\sum_j \|a_j - b_j\|$ | Ano |
| `CosineCoeficient` | $1 - \dfrac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$ | Ne* |

*Kosinová vzdálenost porušuje trojúhelníkovou nerovnost.

Metoda `create_distance_matrix(data)` je **předimplementována** v abstraktní třídě `Distance` — volá `calculate` v cyklu a vrací čtvercovou symetrickou matici $(n \times n)$. Tuto metodu neimplementujete.

> **Klíčový rozdíl oproti Cvičení 02:** V `IterativeClustering` potřebujeme matici vzdáleností **bod–těžiště** tvaru $(n \times k)$, nikoli bod–bod tvaru $(n \times n)$. Těžiště navíc nejsou body datasetu (s výjimkou Forgyho inicializace). Proto `_distances_to_centroids` je samostatná metoda — nelze použít `create_distance_matrix` přímo.

---

## Konfigurace projektu

### Soubor `config.yaml`

Parametry experimentu jsou uloženy v souboru `config.yaml` v kořenovém adresáři:

```yaml
common:
  random_state: 42       # zárodek generátoru pro reprodukovatelnost
  max_iter: 100          # maximální počet iterací obou algoritmů

data:
  image: data/Bunky.png        # vstupní snímek k segmentaci
  # image: data/Bunky_real.png # alternativní snímek — přepnete odkomentováním

kmeans:
  k: 4
  initializer: kmeans++  # možnosti: random_uniform | forgy | kmeans++

fuzzy_cmeans:
  k: 4
  q: 2.0                 # parametr fuzifikace (pouze FCM), musí být > 1
  initializer: random_uniform
```

Vstupní snímek je řízen sekcí `data` — přepnutí na jiný obrázek je jen úprava
konfigurace, ne kódu. Druhá cesta je připravená jako zakomentovaný řádek.

Parametry lze volně měnit bez úpravy kódu. Pokud chcete otestovat jiný počet shluků nebo inicializaci, stačí upravit `config.yaml` a znovu spustit `cviceni_03.py`.

### Typovaná konfigurace (dataclassy)

Konfigurace je načtena funkcí `load_config()` a vrácena jako instance typované dataclassy `ExperimentConfig`. Přístup k parametrům je přes atributy, ne slovníkové klíče:

```
# Místo:  cfg["kmeans"]["k"]     ← runtime chyba při překlepu
# Správně: cfg.kmeans.k          ← editor odhalí překlep okamžitě
```

Struktura dataclassů:

```
ExperimentConfig
    ├── common: CommonConfig
    │       ├── random_state: int
    │       └── max_iter: int
    ├── data: DataConfig
    │       └── image: str
    ├── kmeans: KMeansConfig
    │       ├── k: int
    │       └── initializer: str
    └── fuzzy_cmeans: FuzzyCMeansConfig
            ├── k: int
            ├── q: float
            └── initializer: str
```

Tato hierarchická struktura přímo zrcadlí sekce YAML souboru. Při načítání se klíče každé sekce automaticky mapují na atributy příslušné dataclassy.

> **Proč dataclassy, ne slovník?** Editor (VS Code / PyCharm) zná typy atributů předem. Překlep jako `cfg.kmeens.k` se zobrazí jako červená vlnovka ihned při psaní — bez nutnosti spustit kód. Slovníkový přístup `cfg["kmeens"]["k"]` způsobí `KeyError` až za běhu.

---

## Pokyny k vypracování

Otevřete soubory popsané níže a nahraďte všechny výskyty `raise NotImplementedError(...)` funkčním kódem. Implementujte bloky v pořadí, jak jsou uvedeny — každý blok závisí na předchozím.

Komentáře ve tvaru `# assert: Ověřte, že ...` jsou **nápovědy pro validaci vstupů**. Napište odpovídající příkazy `assert` na daná místa — chrání vás před záhadnými chybami při špatném vstupu.

---

### Předpoklad: třídy vzdálenosti z Cvičení 01 — `src/distance.py`

Pipeline vyžaduje funkční `EuclideanDistance` (i ostatní metriky jsou k implementaci). **Zkopírujte** svoji implementaci z Cvičení 01 do `src/distance.py`. Kostra tříd je připravena — stačí doplnit těla metod `calculate` v každé třídě.

Připomenutí vzorců:

- **Euklidovská:** $d = \sqrt{\sum_j (a_j - b_j)^2}$
- **Manhattanská:** $d = \sum_j |a_j - b_j|$
- **Kosinová:** $d = 1 - \dfrac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$ — ošetřete případ nulové normy

---

### Blok I: Inicializační strategie — `src/initialization.py`

#### `RandomUniformInit.initialize(x, k)`

Vygenerujte $k$ těžišť náhodně z rovnoměrného rozdělení v rozsahu dat.

```
# Pro každý příznak j:
#   najdi minimum a maximum sloupce x[:, j]
#   vygeneruj k náhodných hodnot z intervalu [min, max] pomocí self._rng
# Vrať matici těžišť tvaru (k, n_příznaků)
```

Nápověda: `self._rng.uniform(low=..., high=..., size=...)` nebo `self._rng.random` s manuálním škálováním.

#### `ForgyInit.initialize(x, k)`

Vyberte $k$ různých řádků z $x$ jako počáteční těžiště.

```
# Vyberte k různých indexů z rozsahu 0 .. n-1 bez opakování pomocí self._rng
# Vrať x[vybrané_indexy] — těžiště jsou přímo body datasetu
```

Nápověda: `self._rng.choice(n, size=k, replace=False)` — parametr `replace=False` zajišťuje, že žádný bod nebude vybrán dvakrát.

#### `KMeansPlusPlusInit.initialize(x, k)`

Implementujte algoritmus k-means++:

```
# 1. Vyberte první těžiště náhodně (rovnoměrně) z bodů x
# 2. Opakujte (k-1) krát:
#    a. Pro každý bod vypočítejte vzdálenost k nejbližšímu dosud zvolenému těžišti
#    b. Umocněte vzdálenosti na druhou → D_squared
#    c. Normalizujte na pravděpodobnosti: p = D_squared / sum(D_squared)
#    d. Vyberte další těžiště s pravděpodobností p pomocí self._rng.choice
# 3. Vrať matici k těžišť tvaru (k, n_příznaků)
```

Vzdálenosti v kroku 2a počítejte jako euklidovskou normu (`np.linalg.norm`) — inicializátor je nezávislý na injektované metrice shlukování.

---

### Blok II: Bázová třída — `src/base.py`

Třída `IterativeClustering` je abstraktní (`ABC`). Implementujte v ní tři sdílené metody.

#### `_distances_to_centroids(x, centroids)`

Sestavte obdélníkovou matici vzdáleností bod–těžiště tvaru $(n, k)$:

```
# Inicializujte prázdnou matici tvaru (n_bodů, k)
# Pro každý bod i v rozsahu n_bodů:
#   Pro každé těžiště j v rozsahu k:
#     vypočítejte vzdálenost self.distance.calculate(x[i], centroids[j])
#     uložte do matice na pozici [i, j]
# Vrať matici
```

#### `_has_converged(old_centroids, new_centroids)`

Zkontrolujte, zda se těžiště přestala pohybovat:

```
# Spočítejte normu rozdílu matic: np.linalg.norm(old - new)
# Vraťte True, pokud je norma menší než self._EPSILON (= 1e-6)
```

#### `fit(x)`

Implementujte iterační smyčku — **pořadí kroků je závazné**:

```
# 1. Inicializujte těžiště: self.centroids_ = self.initializer.initialize(x, self.k)
# 2. Opakujte max_iter krát:
#    a. Přiřaďte body k těžištím: assignment = self._update_assignment(x, self.centroids_)
#    b. Uložte stará těžiště pro test konvergence: old = self.centroids_.copy()
#    c. Přepočítejte těžiště: new = self._update_centroids(x, assignment)
#    d. Uložte výsledky: self.assignment_ = assignment; self.centroids_ = new
#    e. Pokud self._has_converged(old, new): zastav smyčku
# 3. Vrať self (umožňuje řetězení km.fit(x).predict())
```

> Přiřazení se počítá ze **starých** těžišť; nová těžiště se počítají z **nového** přiřazení. Záměna pořadí by vedla k nekonzistentnímu stavu, zejména u FCM.

---

### Blok III: K-means — `src/kmeans.py`

#### `KMeans._update_assignment(x, centroids)`

Tvrdé přiřazení — každý bod dostane číslo nejbližšího těžiště:

```
# Vypočítejte matici vzdáleností (n, k): distances = self._distances_to_centroids(x, centroids)
# Pro každý bod najděte index minima podél osy těžišť (axis=1)
# Vrať 1D pole popisků tvaru (n_bodů,) s hodnotami 0 .. k-1
```

#### `KMeans._update_centroids(x, assignment)`

Průměr přiřazených bodů pro každý shluk:

```
# Inicializujte výslednou matici těžišť tvaru (k, n_příznaků)
# Pro každý shluk c v rozsahu 0 .. k-1:
#   Vyberte řádky x kde assignment == c
#   Pokud shluk není prázdný: nové těžiště = průměr těchto řádků (axis=0)
#   Pokud shluk je prázdný: zachovejte staré těžiště self.centroids_[c] (zabraňuje NaN)
# Vrať matici nových těžišť
```

#### `KMeans.predict()`

```
# Ověřte, že fit() byl zavolán (self.assignment_ není None)
# Vrať self.assignment_ — tvrdé popisky uložené v poslední iteraci fit()
```

---

### Blok IV: Fuzzy c-means — `src/fuzzy_cmeans.py`

#### `FuzzyCMeans._update_assignment(x, centroids)`

Matice členství podle vzorce FCM:

```
# Vypočítejte matici vzdáleností (n, k): distances = self._distances_to_centroids(x, centroids)
# Pro každý bod i:
#   Pokud bod leží přesně na těžišti c (distances[i, c] == 0):
#       U[i, c] = 1, ostatní U[i, j] = 0 pro j ≠ c — přeskočte standardní výpočet
#   Jinak pro každé těžiště c:
#       Spočítejte jmenovatele: součet přes všechna j: (d[i,c] / d[i,j])^(2/(q-1))
#       U[i, c] = 1 / jmenovatel
# Vrať matici U tvaru (n_bodů, k), každý řádek sumuje na 1
```

Exponent je `2 / (self.q - 1)`.

#### `FuzzyCMeans._update_centroids(x, assignment)`

Vážený průměr s vahami $U^q$:

```
# Pro každý shluk c:
#   Vezměte sloupec členství: U_c = assignment[:, c]
#   Váhy: weights = U_c ** self.q  (tvar (n_bodů,))
#   Nové těžiště c = součet(weights[:, None] * x, axis=0) / součet(weights)
# Vrať matici těžišť tvaru (k, n_příznaků)
```

#### `FuzzyCMeans.predict()`

```
# Ověřte, že fit() byl zavolán (self.assignment_ není None a je 2D)
# Vrať np.argmax(self.assignment_, axis=1)
# Každý bod dostane číslo shluku s nejvyšším členstvím
```

---

### Blok V: Silhouetová analýza — `src/silhouette.py`

#### `silhouette_samples(x, labels, distance)`

Silhouetová hodnota pro každý bod:

```
# Pro každý bod i:
#   Identifikujte jeho shluk c = labels[i]
#
#   Výpočet a(i):
#     Vyberte body stejného shluku (kromě i samotného)
#     Pokud shluk obsahuje jen bod i → s(i) = 0, přeskočte
#     a(i) = průměrná vzdálenost bodu i od těchto bodů
#
#   Výpočet b(i):
#     Pro každý jiný shluk c' ≠ c:
#         průměrná vzdálenost bodu i od všech bodů shluku c'
#     b(i) = minimum těchto průměrů přes všechna c' ≠ c
#
#   s(i) = (b(i) - a(i)) / max(a(i), b(i))
#
# Vrať pole silhouetových hodnot tvaru (n_bodů,)
```

Pro výpočet každé vzdálenosti volejte `distance.calculate(x[i], x[j])` — konzistentní metrika je klíčová.

#### `silhouette_score(x, labels, distance)`

```
# Zavolejte silhouette_samples(x, labels, distance)
# Vraťte průměr výsledného pole: np.mean(...)
```

---

### Blok VI: Konfigurace — `dataio/config_manager.py`

#### `make_initializer(name, random_state)`

Tovární funkce: převede řetězcový název strategie na instanci `Initializer`.

```
# Vytvořte slovník (registr): název → třída inicializátoru
#   "random_uniform" → RandomUniformInit
#   "forgy"         → ForgyInit
#   "kmeans++"      → KMeansPlusPlusInit
#
# Pokud name není ve slovníku: vyvolejte ValueError s výpisem dostupných možností
# Jinak: vrať registry[name](random_state=random_state)
```

Tato funkce je volána automaticky v `cviceni_03.py` — po implementaci přestane pipeline při spuštění padat na `NotImplementedError`.

#### `validate_config(cfg)`

Ověřte platnost hodnot v konfiguraci:

```
# Ověřte, že cfg.kmeans.k >= 2
# Ověřte, že cfg.fuzzy_cmeans.k >= 2
# Ověřte, že cfg.fuzzy_cmeans.q > 1
# Ověřte, že cfg.kmeans.initializer je jedním z povolených názvů
# Ověřte, že cfg.fuzzy_cmeans.initializer je jedním z povolených názvů
# Pokud cokoliv nevyhoví: vyvolejte ValueError se srozumitelnou zprávou
```

Tato funkce je volána přímo z `load_config` — každé načtení konfigurace tak
projde ověřením a pipeline se nikdy nespustí s neplatným vstupem. Jde o běžný
**obranný vzor**: chráníte i uživatele, kteří nejsou při zadávání důslední.

> Protože `load_config` na `validate_config` závisí, je to jeden z prvních
> úkolů, které je třeba dokončit — dokud není hotový, `cviceni_03.py` se
> korektně ukončí hláškou `[NEDOKONCENO] load_config / validate_config`.
> Testy ostatních komponent (`pytest`) na něm nezávisí a lze je řešit odděleně.

---

### Blok VII: Příprava příznaků — `dataio/features.py`

#### `to_hsv(rgb_data)`

Převeďte příznakovou matici pixelů z prostoru RGB do prostoru HSV:

```
# Vstup: matice tvaru (n_pixelů, 3) s hodnotami v [0, 1] (kanály R, G, B)
# Výstup: matice tvaru (n_pixelů, 3) (kanály H, S, V)
#
# Nápověda: vyhledejte vhodnou knihovní funkci — například:
#   matplotlib.colors.rgb_to_hsv  (pracuje s celou maticí najednou)
#   colorsys.rgb_to_hsv           (pracuje s jedním pixelem; nutné aplikovat po řádcích)
```

Hledání vhodného nástroje je součástí úkolu — obě cesty vedou ke správnému výsledku.

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_03.py -v
```

Testy jsou rozděleny do tříd podle implementované komponenty:

| Třída testů | Co testuje |
|:---|:---|
| `TestInitializers` | Tvar těžišť, rozsah hodnot, reprodukovatelnost seedu, Forgy vybírá body z datasetu |
| `TestBase` | `_distances_to_centroids`: obdélníkový tvar $(n, k)$ a hodnoty; `_has_converged`: prahy |
| `TestKMeans` | Tvar výstupu `predict`, správné rozpoznání tří shluků, chyba bez `fit`, rozsah popisků |
| `TestFuzzyCMeans` | Tvar matice členství, součet členství = 1, tvrdé popisky z `predict`, konvergence |
| `TestSilhouette` | Rozsah hodnot $[-1, 1]$, perfektní shluky mají skóre blízké 1, průměr |
| `TestMakeInitializer` | Továrna vrací správný typ pro každý název, neznámý název vyvolá výjimku |
| `TestConfig` | `validate_config`: platná konfigurace projde, chybné `k`, `q` a název inicializátoru jsou odmítnuty |

Testy používají syntetická data — tři zjevně oddělené shluky, kde správná implementace musí fungovat bez ohledu na inicializaci. Nedeterminismus je ošetřen pevným zárodkem a tolerancí pro permutace popisků (různé číslování shluků je správná odpověď, ne chyba).

Průběžně ověřujte pipeline:

```bash
python cviceni_03.py
```

Pipeline provede celé shlukování na snímku `data/Bunky.png` a zobrazí segmentované obrázky, silhouetové diagramy a křivku skóre pro různá $k$. Kroky s neimplementovanými metodami se přeskočí s výpisem `[NEDOKONCENO]`.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení implementace proveďte:

```bash
git add src/distance.py src/initialization.py src/base.py
git add src/kmeans.py src/fuzzy_cmeans.py src/silhouette.py
git add dataio/features.py dataio/config_manager.py
git commit -m "Implementace cvičení 3"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).

> **Soubory, které se neodevzdávají:** `dataio/__init__.py`, `src/__init__.py`, `dataio/loader.py`, `dataio/plotting.py` a `cviceni_03.py` jsou buď předimplementovány, nebo se nemají měnit. Systém tyto soubory ignoruje a hodnotí pouze výše uvedené.
