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
    Předzpracování příznakových vektorů pixelů.

    Konverze barevného prostoru a výběr kanálů pro experimenty
    s různými reprezentacemi vstupních dat.
"""

import numpy as np
from matplotlib.colors import rgb_to_hsv

def to_hsv(rgb_data: np.ndarray) -> np.ndarray:
    """Převede příznakovou matici z prostoru RGB do prostoru HSV.

    Úkol:
        Implementujte konverzi každého řádku matice ``rgb_data`` z RGB na HSV.
        Vyhledejte vhodnou knihovní funkci — například v ``matplotlib.colors``
        (``rgb_to_hsv``) nebo v modulu ``colorsys`` ze standardní knihovny
        (``colorsys.rgb_to_hsv``). Hledání vhodného nástroje je samo o sobě
        součástí úkolu.

        Vstup má tvar ``(n_pixelů, 3)`` s hodnotami v rozsahu [0, 1].
        Výstup musí mít stejný tvar ``(n_pixelů, 3)`` — kanály H, S, V.

    Parameters
    ----------
    rgb_data:
        Příznakový matice ve formátu RGB, tvar ``(n_pixelů, 3)``,
        hodnoty v rozsahu [0, 1].

    Returns
    -------
    np.ndarray
        Příznakový matice ve formátu HSV, tvar ``(n_pixelů, 3)``.
    """
    assert rgb_data.ndim == 2
    assert rgb_data.shape[1] == 3
    assert np.all((rgb_data >= 0.0) & (rgb_data <= 1.0))

    return rgb_to_hsv(rgb_data)


def select_channels(data: np.ndarray, channels: list[int]) -> np.ndarray:
    """Vybere zadané kanály z příznakové matice.

    Použijte pro experimenty s podmnožinami příznaků, například:
    - ``[0, 1]``  → kanály R a G
    - ``[0, 2]``  → kanály R a B
    - ``[1, 2]``  → kanály G a B

    Parameters
    ----------
    data:
        Příznakový matice tvaru ``(n_pixelů, n_kanálů)``.
    channels:
        Indexy kanálů (sloupců), které mají být zachovány.

    Returns
    -------
    np.ndarray
        Podmatice tvaru ``(n_pixelů, len(channels))``.
    """
    return data[:, channels]
