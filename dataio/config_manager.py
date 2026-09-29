# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Správa konfigurace experimentů ze souboru YAML a továrna inicializátorů.

    Konfigurace je načtena do typovaných dataclass instancí — překlep v názvu
    atributu odhalí editor okamžitě, ne až za běhu.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from src import Initializer


@dataclass
class CommonConfig:
    """Společná nastavení sdílená oběma algoritmy."""

    random_state: int
    max_iter: int


@dataclass
class DataConfig:
    """Nastavení vstupních dat — cesta k segmentovanému snímku."""

    image: str


@dataclass
class KMeansConfig:
    """Nastavení specifická pro algoritmus k-means."""

    k: int
    initializer: str


@dataclass
class FuzzyCMeansConfig:
    """Nastavení specifická pro fuzzy c-means."""

    k: int
    q: float
    initializer: str


@dataclass
class ExperimentConfig:
    """Kompletní konfigurace experimentu načtená z YAML souboru."""

    common: CommonConfig
    data: DataConfig
    kmeans: KMeansConfig
    fuzzy_cmeans: FuzzyCMeansConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Načte konfiguraci experimentu ze souboru YAML.

    Vrací typovanou instanci ``ExperimentConfig`` — přístup přes atributy
    (``cfg.kmeans.k``) místo slovníkových klíčů (``cfg["kmeans"]["k"]``).

    Před vrácením je konfigurace ověřena funkcí ``validate_config`` — při
    neplatných hodnotách je vyvolána výjimka, takže volající vždy dostane
    buď platnou konfiguraci, nebo srozumitelnou chybu.

    Parameters
    ----------
    filepath:
        Cesta k souboru YAML (výchozí: ``config.yaml`` v pracovním adresáři).

    Returns
    -------
    ExperimentConfig
        Typovaná a ověřená konfigurace experimentu.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    cfg = ExperimentConfig(
        common=CommonConfig(**raw["common"]),
        data=DataConfig(**raw["data"]),
        kmeans=KMeansConfig(**raw["kmeans"]),
        fuzzy_cmeans=FuzzyCMeansConfig(**raw["fuzzy_cmeans"]),
    )
    validate_config(cfg)
    return cfg


def make_initializer(name: str, random_state: int | None = None) -> Initializer:
    """Továrna: převede jméno strategie inicializace na instanci ``Initializer``.

    Úkol:
        Vytvořte mapování názvů strategií na třídy inicializátorů a vraťte
        instanci příslušné třídy inicializovanou s daným ``random_state``.

        Podporované názvy (viz ``config.yaml``):
        - ``"random_uniform"``  → ``RandomUniformInit``
        - ``"forgy"``           → ``ForgyInit``
        - ``"kmeans++"``        → ``KMeansPlusPlusInit``

        Pro neznámý název vyvolejte ``ValueError`` s opisem dostupných možností.

        Vzor (registr tříd):
        ::

            registry = {
                "random_uniform": RandomUniformInit,
                "forgy": ForgyInit,
                "kmeans++": KMeansPlusPlusInit,
            }
            if name not in registry:
                raise ValueError(f"Neznámá inicializační strategie: {name}")
            return registry[name](random_state=random_state)

    Parameters
    ----------
    name:
        Název inicializační strategie — čteno z ``config.yaml``.
    random_state:
        Zárodek generátoru náhodných čísel pro reprodukovatelnost.

    Returns
    -------
    Initializer
        Instance vybrané inicializační strategie.
    """
    raise NotImplementedError(
        "Úkol: implementujte funkci make_initializer — namapujte řetězcový "
        "název strategie na instanci příslušné třídy Initializer."
    )


def validate_config(cfg: ExperimentConfig) -> None:
    """Ověří platnost konfigurace a vyvolá výjimku při chybné hodnotě.

    Úkol:
        Přidejte ověření, že konfigurace obsahuje požadované hodnoty
        ve správných rozsazích:
        - ``cfg.kmeans.k >= 2`` a ``cfg.fuzzy_cmeans.k >= 2``
          (shlukování s méně než dvěma shluky nedává smysl)
        - ``cfg.fuzzy_cmeans.q > 1``  (parametr fuzifikace musí být větší než 1)
        - ``cfg.kmeans.initializer`` i ``cfg.fuzzy_cmeans.initializer`` jsou
          jedním z povolených názvů (``random_uniform``, ``forgy``, ``kmeans++``)

        Používejte ``assert`` nebo ``ValueError`` pro srozumitelné chybové zprávy.

        Tato funkce je volána z ``load_config`` — každé načtení konfigurace
        tak projde ověřením a pipeline se nikdy nespustí s neplatným vstupem.
        Jde o běžný obranný vzor: chráníte i uživatele, kteří nejsou při
        zadávání důslední.

    Parameters
    ----------
    cfg:
        Typovaná konfigurace sestavená funkcí ``load_config``.

    Raises
    ------
    ValueError
        Pokud kterákoli hodnota v konfiguraci nesplňuje uvedené podmínky.
    """
    raise NotImplementedError(
        "Úkol: implementujte validate_config — ověřte platnost konfigurace "
        "(k >= 2, q > 1, platný název inicializátoru)."
    )
