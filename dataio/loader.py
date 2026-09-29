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
    Načítání obrazových dat pro cvičení nehierarchického shlukování.

    Každý pixel vstupního obrázku se stane jedním řádkem příznakové matice.
    Výsledná matice má tvar (počet_pixelů, 3), kde sloupce odpovídají kanálům R, G, B.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_image(filepath: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Načte obrázek ze souboru a převede jej na příznakovou matici pixelů.

    Obrázek je načten jako pole float32 s hodnotami v rozsahu [0, 1].
    Každý pixel tvoří jeden řádek matice — sloupce jsou R, G, B.
    Původní rozměry (výška, šířka) jsou vráceny zvlášť, aby bylo možné
    zpětně rekonstruovat segmentační obrázek z plochého vektoru popisků.

    Parameters
    ----------
    filepath:
        Cesta k souboru PNG (nebo jiný formát podporovaný matplotlib).

    Returns
    -------
    rgb_data:
        Pole tvaru ``(n_pixelů, 3)`` — jeden řádek na pixel, kanály R G B.
    dimensions:
        Dvojice ``(výška, šířka)`` potřebná pro zpětné přetvarování popisků.
    """
    image: np.ndarray = plt.imread(filepath).astype(np.float32)

    # Pokud má obrázek alfa kanál, odřízneme ho
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    height: int = image.shape[0]
    width: int = image.shape[1]

    # Přetvarujeme (výška, šířka, 3) → (výška*šířka, 3)
    rgb_data: np.ndarray = image.reshape(-1, 3)

    return rgb_data, (height, width)
