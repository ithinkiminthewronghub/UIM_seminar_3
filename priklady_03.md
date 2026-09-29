# Příklady 03 – Nehierarchické shlukování

Všechny úlohy jsou navrženy pro ruční výpočet ve 2D prostoru. Vzdálenosti jsou
euklidovské: d(x, y) = √[(x₁−y₁)² + (x₂−y₂)²].
Pro přiřazení k shlukům stačí porovnávat **čtvercové** vzdálenosti d² (odpadá
výpočet odmocniny, výsledek je stejný).

---

## Příklad 1 – K-means: první iterace

**Data** — devět bodů ve 2D:

| Bod | x₁  | x₂  |
|-----|-----|-----|
| P1  | −14 |  10 |
| P2  | −12 |   6 |
| P3  | −10 |   8 |
| P4  |   0 | −10 |
| P5  |   4 |  −8 |
| P6  |   2 |  −6 |
| P7  |  12 |   4 |
| P8  |  16 |   2 |
| P9  |  14 |   6 |

Počet shluků **k = 3**. Forgyho inicializace vybrala jako počáteční těžiště
tři existující body:

c₁ = P1 = (−14, 10),  c₂ = P4 = (0, −10),  c₃ = P9 = (14, 6)

**Úkoly:**

1. Pro každý bod vypočítejte čtvercovou vzdálenost ke všem třem těžištím:
   d²(x, c) = (x₁ − c₁)² + (x₂ − c₂)²

2. Přiřaďte každý bod do shluku s nejmenší hodnotou d².

3. Vypočítejte nová těžiště jako aritmetický průměr souřadnic bodů v každém
   shluku.

---

## Příklad 2 – K-means: konvergence

Navazuje na Příklad 1. Nová těžiště po první iteraci jsou:

c₁ = (−12, 8),  c₂ = (2, −8),  c₃ = (14, 4)

**Úkoly:**

1. Zopakujte krok přiřazení pro 2. iteraci s novými těžišti.

2. Jsou výsledné shluky stejné jako po 1. iteraci? Jaký závěr z toho plyne?

3. Jak se posunula těžiště oproti konci 1. iterace? Jak tuto změnu změří
   algoritmus v kritériu konvergence?

---

## Příklad 3 – K-means: lokální minimum a vliv inicializace

**Data** — šest bodů ve 2D:

| Bod | x₁  | x₂ |
|-----|-----|----|
| Q1  | −12 | 10 |
| Q2  |  −8 | 10 |
| Q3  |   0 |  −2 |
| Q4  |   0 |   6 |
| Q5  |  14 |   0 |
| Q6  |  14 |   8 |

Počet shluků **k = 3**.

**Špatná inicializace:** c₁ = Q1 = (−12, 10),  c₂ = Q2 = (−8, 10),  c₃ = Q3 = (0, −2)

**Dobrá inicializace:**  c₁ = Q1 = (−12, 10),  c₂ = Q3 = (0, −2),   c₃ = Q5 = (14, 0)

**Úkoly:**

1. Proveďte 1. iteraci k-means pro **špatnou** inicializaci:
   přiřaďte body a vypočítejte nová těžiště.

2. Ověřte, zda algoritmus konvergoval po 2. iteraci (proveďte 2. iteraci a
   srovnejte přiřazení).

3. Proveďte 1. iteraci pro **dobrou** inicializaci:
   přiřaďte body a vypočítejte nová těžiště.

4. Vypočítejte **celkovou inercii** pro každé řešení:
   inertia = Σᵢ d²(xᵢ, c_{k(i)}), kde c_{k(i)} je těžiště přiřazeného shluku.

5. Jak se liší výsledky? Co říká nižší inertia o kvalitě řešení?

---

## Příklad 4 – K-means++: výběr počátečních těžišť

Stejná data jako v Příkladu 3 (Q1–Q6), **k = 3**.
Algoritmus k-means++ zvolil jako první těžiště **c₁ = Q1 = (−12, 10)**.

**Úkoly:**

1. Pro každý bod Q1–Q6 vypočítejte D²(x) = d²(x, c₁).

2. Normalizujte hodnoty D²(x) na pravděpodobnosti p(x) = D²(x) / ΣD²(x).
   Které těžiště bude pravděpodobně zvoleno jako c₂? (Bod s nejvyšší p.)

3. Předpokládejte, že algoritmus vybral **c₂ = Q5 = (14, 0)**.
   Aktualizujte D²(x) jako minimum vzdáleností k oběma dosud zvoleným těžištím:
   D²(x) ← min(d²(x, c₁), d²(x, c₂))

4. Znovu normalizujte na pravděpodobnosti. Které těžiště bude zvoleno jako c₃?

5. Srovnejte tři těžiště zvolená k-means++ s „špatnou inicializací"
   z Příkladu 3. Co k-means++ dělá jinak?

---

## Příklad 5 – Fuzzy c-means: matice členství

**Data:** dva body, **k = 2**, parametr fuzifikace **q = 2**:

- P1 = (−6, 8)
- P2 = (6, −8)

**Počáteční těžiště:** c₁ = (−3, 4),  c₂ = (3, −4)

**Úkoly:**

1. Vypočítejte euklidovské vzdálenosti:
   d(P1, c₁),  d(P1, c₂),  d(P2, c₁),  d(P2, c₂)

2. Pomocí vzorce FCM vypočítejte míru členství U(xᵢ, cⱼ) pro q = 2:

$$U(x_i,\, c_j) \;=\; \frac{1}{\displaystyle\sum_{l=1}^{k} \!\left(\frac{d(x_i,\, c_j)}{d(x_i,\, c_l)}\right)^{\!\!\frac{2}{q-1}}}$$

   Pro q = 2 platí exponent 2/(q−1) = 2.

3. Sestavte matici členství U tvaru (2 × 2).

4. Ověřte, že každý řádek matice U se sčítá na 1.

---

## Příklad 6 – Fuzzy c-means: přepočet těžišť

Navazuje na Příklad 5. Matice členství z předchozí iterace:

| Bod | U(·, c₁) | U(·, c₂) |
|-----|----------|----------|
| P1  |   9/10   |   1/10   |
| P2  |   1/10   |   9/10   |

**Úkoly:**

1. Vypočítejte nová těžiště pomocí váženého průměru. Váhy jsou U^q = U²:

$$c_j^{\text{nové}} \;=\; \frac{\displaystyle\sum_{i} U(x_i,\, c_j)^{q} \cdot x_i}{\displaystyle\sum_{i} U(x_i,\, c_j)^{q}}$$

2. Porovnejte nová těžiště se starými (c₁ = (−3, 4), c₂ = (3, −4)).
   Jakým směrem se každé těžiště posunulo? Odpovídá to hodnotám v matici U?

---

## Příklad 7 – Silhouetová analýza pro k = 2

**Data:** šest bodů, rozdělených do dvou shluků:

- C₁ = { S1 = (0, 0),   S2 = (6, 0),   S3 = (3, 4)  }
- C₂ = { S4 = (15, 0),  S5 = (18, 0),  S6 = (15, 4) }

Pro urychlení jsou všechny vzdálenosti předpočítány (zaokrouhleno na 1 desetinné místo):

|    | S1   | S2   | S3   | S4   | S5   | S6   |
|----|------|------|------|------|------|------|
| S1 |  —   |  6,0 |  5,0 | 15,0 | 18,0 | 15,5 |
| S2 |  6,0 |  —   |  5,0 |  9,0 | 12,0 |  9,8 |
| S3 |  5,0 |  5,0 |  —   | 12,6 | 15,5 | 12,0 |
| S4 | 15,0 |  9,0 | 12,6 |  —   |  3,0 |  4,0 |
| S5 | 18,0 | 12,0 | 15,5 |  3,0 |  —   |  5,0 |
| S6 | 15,5 |  9,8 | 12,0 |  4,0 |  5,0 |  —   |

**Silhouetový koeficient** pro bod x přiřazený do shluku A:
- a(x) = průměrná vzdálenost ke všem ostatním bodům v A
- b(x) = průměrná vzdálenost ke všem bodům v nejbližším jiném shluku B
- s(x) = (b − a) / max(a, b),  s(x) ∈ [−1, 1]

**Úkoly:**

1. Pro bod **S1** (shluk C₁) vypočítejte a(S1), b(S1) a s(S1).

2. Stejný postup pro body S2, S3 (shluk C₁) a S4, S5, S6 (shluk C₂).

3. Vypočítejte průměrné silhouetové skóre pro toto dělení (k = 2).

---

## Příklad 8 – Výběr počtu shluků: k = 2 vs. k = 3

Stejná data a stejná vzdálenostní tabulka jako v Příkladu 7.
Tentokrát uvažujeme dělení pro **k = 3**:

- C₁ = { S1, S2 }
- C₂ = { S3 }  ← singleton
- C₃ = { S4, S5, S6 }

_Poznámka:_ Pro bod v singletonovém shluku platí s = 0 (konvence — a není definováno).

**Úkoly:**

1. Vypočítejte a(x), b(x) a s(x) pro všechny body při k = 3.
   Pro b(x): průměrná vzdálenost k bodům v **nejbližším** jiném shluku.

2. Vypočítejte průměrné silhouetové skóre pro k = 3.

3. Porovnejte skóre pro k = 2 a k = 3. Které dělení je podle silhouety lepší?

4. Proč mají body S1 a S2 záporné silhouetové hodnoty při k = 3?
   Co nám to říká o kvalitě tohoto dělení?
