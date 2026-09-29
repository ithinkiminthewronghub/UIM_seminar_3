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
    Algoritmus k-means jako podtřída IterativeClustering.

    Tři krátké přepisy variačních bodů — iterační smyčka zůstává v základní třídě.
"""

from __future__ import annotations

import numpy as np

from src.base import IterativeClustering


class KMeans(IterativeClustering):
    """K-means shlukování s tvrdým přiřazením bodů k nejbližšímu těžišti.

    Dědí kompletní iterační smyčku z ``IterativeClustering``.
    Implementuje pouze:
    - přiřazení bodů jako ``argmin`` vzdáleností (tvrdé popisky),
    - přepočet těžišť jako prostý aritmetický průměr přiřazených bodů.
    """

    def _update_assignment(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Přiřadí každý bod k nejbližšímu těžišti (tvrdé přiřazení).

        Úkol:
            1. Vypočítejte matici vzdáleností bod-těžiště pomocí
               ``self._distances_to_centroids(x, centroids)``.
            2. Pro každý bod najděte index nejbližšího těžiště pomocí
               ``np.argmin`` podél osy těžišť (``axis=1``).

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        centroids:
            Aktuální těžiště tvaru ``(k, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.
        """
        # assert: Ověřte, že x a centroids jsou 2D matice se stejným počtem příznaků
        raise NotImplementedError(
            "Úkol: implementujte KMeans._update_assignment() — použijte "
            "_distances_to_centroids a np.argmin pro tvrdé přiřazení."
        )

    def _update_centroids(
        self, x: np.ndarray, assignment: np.ndarray
    ) -> np.ndarray:
        """Přepočítá těžiště jako prostý průměr přiřazených bodů.

        Úkol:
            Pro každý shluk ``c`` v rozsahu 0 … k-1:
            1. Vyberte řádky ``x``, kde ``assignment == c``.
            2. Spočítejte průměr ``np.mean(..., axis=0)``.
            3. Pokud je shluk prázdný (žádný bod nepatří do ``c``), zachovejte
               staré těžiště ``self.centroids_[c]`` — zabraňuje NaN.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        assignment:
            Tvrdé popisky z ``_update_assignment``, tvar ``(n_bodů,)``.

        Returns
        -------
        np.ndarray
            Nová těžiště tvaru ``(k, n_příznaků)``.
        """
        # assert: Ověřte, že assignment je 1D pole a jeho délka odpovídá počtu bodů v x
        raise NotImplementedError(
            "Úkol: implementujte KMeans._update_centroids() — průměr bodů "
            "přiřazených ke každému shluku. Ošetřete prázdné shluky."
        )

    def predict(self) -> np.ndarray:
        """Vrátí tvrdé popisky shluků uložené po volání ``fit``.

        Úkol:
            Vraťte ``self.assignment_`` — pole tvaru ``(n_bodů,)``
            s hodnotami 0 … k-1 přiřazenými v poslední iteraci ``fit``.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``.

        Raises
        ------
        RuntimeError
            Pokud ``fit`` nebyl dosud volán.
        """
        # assert: Ověřte, že fit() byl zavolán (self.assignment_ není None)
        raise NotImplementedError(
            "Úkol: implementujte KMeans.predict() — vraťte self.assignment_ "
            "(tvrdé popisky uložené metodou fit)."
        )
