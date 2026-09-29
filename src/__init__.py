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
    Balíček src - algoritmy nehierarchického shlukování pro Cvičení 03.

    Veřejné API:
    Distance            - abstraktní základ pro metriky vzdálenosti
    EuclideanDistance   - euklidovská metrika
    ManhattanDistance   - manhattanská metrika
    CosineCoeficient    - kosinová podobnost
    Initializer         - abstraktní základ pro inicializační strategie
    RandomUniformInit   - inicializace rovnoměrným rozdělením (vzorový příklad)
    ForgyInit           - Forgyho metoda (stub - k implementaci)
    KMeansPlusPlusInit  - k-means++ (stub - k implementaci)
    IterativeClustering - abstraktní základ se šablonovou metodou (Template Method)
    KMeans              - algoritmus k-means (stub - k implementaci)
    FuzzyCMeans         - fuzzy c-means (stub - k implementaci)
    silhouette_samples  - silhouetové koeficienty pro každý bod
    silhouette_score    - průměrné silhouetové skóre
"""

from src.distance import Distance, EuclideanDistance, ManhattanDistance, CosineCoeficient
from src.initialization import Initializer, RandomUniformInit, ForgyInit, KMeansPlusPlusInit
from src.base import IterativeClustering
from src.kmeans import KMeans
from src.fuzzy_cmeans import FuzzyCMeans
from src.silhouette import silhouette_samples, silhouette_score

__all__ = [
    "Distance",
    "EuclideanDistance",
    "ManhattanDistance",
    "CosineCoeficient",
    "Initializer",
    "RandomUniformInit",
    "ForgyInit",
    "KMeansPlusPlusInit",
    "IterativeClustering",
    "KMeans",
    "FuzzyCMeans",
    "silhouette_samples",
    "silhouette_score",
]
