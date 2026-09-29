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
    Balíček dataio - vstup/výstup a vizualizace pro Cvičení 03.

    Veřejné API:
    ExperimentConfig    - typovaná konfigurace experimentu
    CommonConfig        - společná nastavení (random_state, max_iter)
    DataConfig          - nastavení vstupních dat (cesta k obrázku)
    KMeansConfig        - nastavení k-means (k, initializer)
    FuzzyCMeansConfig   - nastavení FCM (k, q, initializer)
    load_image          - načte PNG jako matici pixelů (n, 3)
    to_hsv              - převede RGB data na HSV
    select_channels     - vybere podmnožinu příznaků
    load_config         - načte konfiguraci z YAML souboru → ExperimentConfig
    make_initializer    - továrna: řetězec → instance Initializer
    validate_config     - ověří platnost konfiguračního slovníku
    plot_segmentation   - zobrazí segmentovaný obrázek
    plot_silhouette     - silhouetový diagram
    plot_k_selection    - křivka silhouetového skóre vs. k
"""

from dataio.loader import load_image
from dataio.features import to_hsv, select_channels
from dataio.config_manager import (
    ExperimentConfig,
    CommonConfig,
    DataConfig,
    KMeansConfig,
    FuzzyCMeansConfig,
    load_config,
    make_initializer,
    validate_config,
)
from dataio.plotting import plot_segmentation, plot_silhouette, plot_k_selection

__all__ = [
    "ExperimentConfig",
    "CommonConfig",
    "DataConfig",
    "KMeansConfig",
    "FuzzyCMeansConfig",
    "load_image",
    "to_hsv",
    "select_channels",
    "load_config",
    "make_initializer",
    "validate_config",
    "plot_segmentation",
    "plot_silhouette",
    "plot_k_selection",
]
