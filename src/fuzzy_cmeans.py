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
    Algoritmus fuzzy c-means jako podtřída IterativeClustering.

    Klíčový rozdíl oproti k-means: bod patří do všech shluků zároveň s různou
    měrou členství. Parametr ``q`` řídí míru „rozmazání" hranic shluků.
"""

from __future__ import annotations

import numpy as np

from src import (
    IterativeClustering,
    Distance,
    Initializer,
    )


class FuzzyCMeans(IterativeClustering):
    """Fuzzy c-means shlukování s měkkým přiřazením bodů.

    Každý bod má přiřazen vektor členství délky ``k``, jehož složky udávají
    míru příslušnosti do každého shluku. Součet členství každého bodu je 1.

    Parametr ``q`` (fuzzifikace) ovlivňuje míru překryvu shluků:
    - ``q → 1``:  přiblíží se k-means (tvrdé hranice),
    - ``q = 2``:  standardní fuzzy c-means,
    - ``q > 2``:  velmi měkké hranice (shluky se silně překrývají).
    """

    def __init__(
        self,
        k: int,
        distance: Distance,
        initializer: Initializer,
        q: float = 2.0,
        max_iter: int = 100,
    ) -> None:
        """Inicializuje FCM s parametrem fuzifikace ``q``.

        Parameters
        ----------
        k:
            Počet shluků.
        distance:
            Instance metriky vzdálenosti (injektovaná závislost).
        initializer:
            Instance inicializační strategie (injektovaná závislost).
        q:
            Parametr fuzifikace — musí být > 1. Výchozí hodnota 2.0
            odpovídá standardnímu fuzzy c-means.
        max_iter:
            Maximální počet iterací.
        """
        # assert: Ověřte, že parametr fuzifikace q je větší než 1
        super().__init__(k, distance, initializer, max_iter)
        self.q: float = q

    def _update_assignment(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Vypočítá matici členství pomocí standardního vzorce FCM.

        Úkol:
            Implementujte výpočet matice členství ``U`` tvaru ``(n_bodů, k)``:

            Pro každý bod ``i`` a shluk ``c``:

            .. math::
                U_{ic} = \\frac{1}{\\sum_{j=1}^{k}
                \\left(\\frac{d_{ic}}{d_{ij}}\\right)^{\\frac{2}{q-1}}}

            kde :math:`d_{ic}` je vzdálenost bodu ``i`` od těžiště ``c``.

            Každý řádek ``U[i]`` musí sumovat na 1.

            Pozor na dělení nulou:
            Pokud bod leží přesně na těžišti (``d_{ic} = 0``), přiřaďte
            tomuto bodu plné členství v daném shluku (``U[i, c] = 1``,
            ostatní složky = 0) a přeskočte standardní výpočet.

            Nápověda: exponent je ``2 / (self.q - 1)``.

        Parameters
        ----------
        X:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        centroids:
            Aktuální těžiště tvaru ``(k, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Matice členství tvaru ``(n_bodů, k)``, každý řádek sumuje na 1.
        """
        # assert: Ověřte, že x a centroids jsou 2D matice se stejným počtem příznaků
        assert x.ndim == 2
        assert centroids.ndim == 2
        assert x.shape[1] == centroids.shape[1]

        n_samples = x.shape[0]
        distances = self._distances_to_centroids(x, centroids)
        matrix_u = np.zeros((n_samples, self.k))
        exponent = 2/(self.q - 1)

        for i in range(n_samples):
            # Ošetření případu, kdy bod leží přesně na těžišti (vzdálenost == 0)
            zero_dist_indices = np.where(distances[i] == 0)[0]
            if len(zero_dist_indices) > 0:
                matrix_u[i, zero_dist_indices] = 1.0 / len(zero_dist_indices)
            else:
                for c in range(self.k):
                    ratio = distances[i, c] / distances[i]
                    matrix_u[i, c] = 1.0 / np.sum(ratio ** exponent)

        return matrix_u


    def _update_centroids(
        self, x: np.ndarray, assignment: np.ndarray
    ) -> np.ndarray:
        """Přepočítá těžiště jako vážený průměr bodů s vahami z matice členství.

        Úkol:
            Pro každý shluk ``c`` spočítejte vážený průměr:

            .. math::
                \\text{centroid}_c =
                \\frac{\\sum_{i} U_{ic}^q \\cdot X_i}{\\sum_{i} U_{ic}^q}

            kde ``U[:, c]`` je sloupec matice členství pro shluk ``c``
            a váhy jsou ``U[:, c] ** self.q``.

            Nápověda: váhy mají tvar ``(n_bodů,)`` — pro vážený průměr
            rozměrů ``(n_příznaků,)`` je třeba správné přetvarování.

        Parameters
        ----------
        X:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        assignment:
            Matice členství tvaru ``(n_bodů, k)`` — výstup ``_update_assignment``.

        Returns
        -------
        np.ndarray
            Nová těžiště tvaru ``(k, n_příznaků)``.
        """
        assert assignment.ndim == 2
        assert assignment.shape == (x.shape[0], self.k)

        n_features = x.shape[1]
        new_centroids = np.zeros((self.k, n_features), dtype=np.float64)

        # Váhy U_{ic}^q pro všechny body a shluky
        u_q = assignment ** self.q

        for c in range(self.k):
            weights = u_q[:, c]
            sum_weights = np.sum(weights)

            if sum_weights > 0:
                new_centroids[c] = np.sum(x * weights[:, np.newaxis], axis=0) / sum_weights
            else:
                assert self.centroids_ is not None
                new_centroids[c] = self.centroids_[c]

        return new_centroids



    def predict(self) -> np.ndarray:
        """Převede měkkou matici členství na tvrdé popisky shluků (argmax).

        Úkol:
            Vraťte ``np.argmax(self.assignment_, axis=1)`` — pro každý bod
            vyberte shluk s nejvyšším členstvím.

            Tím jsou výstupy FCM kompatibilní s ``silhouette.py``, které
            očekává tvrdé popisky stejně jako k-means.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.

        Raises
        ------
        RuntimeError
            Pokud ``fit`` nebyl dosud volán.
        """
        # assert: Ověřte, že fit() byl zavolán a self.assignment_ je 2D matice členství
        if self.assignment_ is None:
            raise RuntimeError("Metoda fit() musí být zavolána před voláním predict().")
        assert self.assignment_.ndim == 2
        return np.argmax(self.assignment_, axis=1)

