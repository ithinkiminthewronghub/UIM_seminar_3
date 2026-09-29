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
    Abstraktní třída vzdálenosti a její konkrétní implementace.

    Zkopírujte sem své řešení z Cvičení 01 — bez funkční metody ``calculate()``
    se shlukování nerozjede. Toto je záměrná kontinuita: stejná třída ``Distance``
    propojuje Cvičení 01, 02 a 03.
"""

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """Abstraktní základ pro metriky vzdálenosti.

    Každá konkrétní metrika dědí od této třídy a implementuje metodu
    ``calculate``. Metoda ``create_distance_matrix`` je sdílená a volá
    ``calculate`` v cyklu — není třeba ji přepisovat.
    """

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí ``True``, pokud vzdálenost splňuje axiomy metriky."""

    @abstractmethod
    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá vzdálenost mezi dvěma body.

        Úkol:
            Implementujte výpočet vzdálenosti specifický pro danou metriku.

        Parameters
        ----------
        point_a:
            První bod — pole tvaru ``(n_příznaků,)``.
        point_b:
            Druhý bod — pole tvaru ``(n_příznaků,)``.

        Returns
        -------
        float
            Vzdálenost mezi ``point_a`` a ``point_b``.
        """
        # assert: Ověřte, že oba vstupy jsou 1D vektory stejné délky
        raise NotImplementedError(
            "Úkol: implementujte calculate() — zkopírujte řešení z Cvičení 01."
        )

    def create_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Vytvoří čtvercovou matici vzdáleností mezi všemi dvojicemi bodů.

        Matice je symetrická s nulovou diagonálou.
        Volá ``self.calculate`` pro každou dvojici — implementace metriky
        není třeba zde duplikovat.

        Parameters
        ----------
        data:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Matice vzdáleností tvaru ``(n_bodů, n_bodů)``.
        """
        # assert: Ověřte, že data jsou 2D matice s alespoň 2 body
        n: int = data.shape[0]
        matrix: np.ndarray = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dist: float = self.calculate(data[i], data[j])
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix


class EuclideanDistance(Distance):
    """Euklidovská vzdálenost — délka přímé spojnice dvou bodů."""

    @property
    def is_metric(self) -> bool:
        """Euklidovská vzdálenost je pravá metrika."""
        raise NotImplementedError(
            "Úkol: implementujte EuclideanDistance.is_metric() — zkopírujte z Cvičení 01."
        )

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá euklidovskou vzdálenost mezi dvěma body.

        Úkol:
            Implementujte vzorec: ``sqrt(sum((a_i - b_i)^2))``.
            Zkopírujte řešení z Cvičení 01.

        Parameters
        ----------
        point_a:
            První bod.
        point_b:
            Druhý bod.

        Returns
        -------
        float
            Euklidovská vzdálenost.
        """
        # assert: Ověřte, že oba vstupy jsou 1D vektory stejné délky
        raise NotImplementedError(
            "Úkol: implementujte EuclideanDistance.calculate() — zkopírujte z Cvičení 01."
        )


class ManhattanDistance(Distance):
    """Manhattanská vzdálenost — součet absolutních rozdílů souřadnic."""

    @property
    def is_metric(self) -> bool:
        """Manhattanská vzdálenost je pravá metrika."""
        raise NotImplementedError(
            "Úkol: implementujte ManhattanDistance.is_metric() — zkopírujte z Cvičení 01."
        )

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá manhattanskou vzdálenost mezi dvěma body.

        Úkol:
            Implementujte vzorec: ``sum(|a_i - b_i|)``.
            Zkopírujte řešení z Cvičení 01.

        Parameters
        ----------
        point_a:
            První bod.
        point_b:
            Druhý bod.

        Returns
        -------
        float
            Manhattanská vzdálenost.
        """
        # assert: Ověřte, že oba vstupy jsou 1D vektory stejné délky
        raise NotImplementedError(
            "Úkol: implementujte ManhattanDistance.calculate() — zkopírujte z Cvičení 01."
        )


class CosineCoeficient(Distance):
    """Kosinová podobnost (jako vzdálenost: 1 - kosinová_podobnost)."""

    @property
    def is_metric(self) -> bool:
        """Kosinová vzdálenost není pravá metrika (porušuje trojúhelníkovou nerovnost)."""
        raise NotImplementedError(
            "Úkol: implementujte CosineCoeficient.is_metric() — zkopírujte z Cvičení 01."
        )

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Vypočítá kosinovou vzdálenost mezi dvěma body.

        Úkol:
            Implementujte vzorec: ``1 - (a · b) / (||a|| * ||b||)``.
            Ošetřete případ, kdy je norma jednoho z vektorů nulová.
            Zkopírujte řešení z Cvičení 01.

        Parameters
        ----------
        point_a:
            První bod.
        point_b:
            Druhý bod.

        Returns
        -------
        float
            Kosinová vzdálenost v rozsahu [0, 2].
        """
        # assert: Ověřte, že oba vstupy jsou 1D vektory stejné délky
        raise NotImplementedError(
            "Úkol: implementujte CosineCoeficient.calculate() — zkopírujte z Cvičení 01."
        )
