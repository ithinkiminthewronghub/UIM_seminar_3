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
    Abstraktní základ pro iterativní nehierarchické shlukování.

    Implementuje šablonovou metodu (Template Method): iterační smyčka je sdílená,
    dva variační body (přiřazení a přepočet těžišť) implementují podtřídy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src import (
        Distance,
        Initializer,
    )


class IterativeClustering(ABC):
    """Abstraktní základ pro k-means a fuzzy c-means.

    Sdílí kompletní iterační smyčku (inicializace → přiřazení → přepočet →
    konvergence). Podtřídy implementují pouze dva variační body:
    ``_update_assignment`` a ``_update_centroids``.

    ``distance`` a ``initializer`` jsou **injektované závislosti** — třída
    nezná a nevybírá konkrétní implementaci.
    """

    _EPSILON: float = 1e-6
    """Práh konvergence pro posun těžišť (pevná třídní konstanta, ne parametr)."""

    def __init__(
        self,
        k: int,
        distance: Distance,
        initializer: Initializer,
        max_iter: int = 100,
    ) -> None:
        """Inicializuje shlukování s injektovanými závislostmi.

        Parameters
        ----------
        k:
            Počet shluků (musí být >= 2).
        distance:
            Instance metriky vzdálenosti (injektovaná závislost).
        initializer:
            Instance inicializační strategie (injektovaná závislost).
        max_iter:
            Maximální počet iterací před nuceným zastavením.
        """
        self.k: int = k
        self.distance: Distance = distance
        self.initializer: Initializer = initializer
        self.max_iter: int = max_iter
        self.centroids_: np.ndarray | None = None
        self.assignment_: np.ndarray | None = None

    def _distances_to_centroids(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Vypočítá obdélníkovou matici vzdáleností bod-těžiště.

        Úkol:
            Implementujte výpočet matice vzdáleností tvaru ``(n_bodů, k)``,
            kde prvek ``[i, j]`` je vzdálenost bodu ``x[i]`` od těžiště
            ``centroids[j]``. Volejte ``self.distance.calculate(x[i], centroids[j])``
            ve dvou vnořených cyklech.

            Zamyslete se: v čem se tato matice liší od ``create_distance_matrix``
            z Cvičení 02?
            → Zde je matice **obdélníková** (m x k): body vs. těžiště.
            → V Cvičení 02 byla matice **čtvercová** (n x n): body vs. body.
            Tato asymetrie odráží podstatu iterativního shlukování — těžiště
            nejsou body datasetu (kromě Forgyho inicializace).

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        centroids:
            Matice těžišť tvaru ``(k, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Matice vzdáleností tvaru ``(n_bodů, k)``.
        """
        # assert: Ověřte, že x a centroids jsou 2D matice se stejným počtem příznaků
        # a centroids má self.k řádků
        raise NotImplementedError(
            "Úkol: implementujte _distances_to_centroids() — vytvořte matici "
            "vzdáleností tvaru (n_bodů, k) voláním self.distance.calculate."
        )

    def _has_converged(
        self,
        old_centroids: np.ndarray,
        new_centroids: np.ndarray,
    ) -> bool:
        """Zkontroluje, zda se těžiště přestala pohybovat (konvergence).

        Úkol:
            Implementujte kritérium konvergence: vraťte ``True``, pokud je
            norma rozdílu starých a nových těžišť menší než ``self._EPSILON``.

            Nápověda: ``np.linalg.norm(old_centroids - new_centroids)``

            Proč posun těžišť, ne změna přiřazení?
            Toto jednotné kritérium funguje pro k-means i fuzzy c-means —
            v FCM jsou „přiřazení" měkké (matice členství), ale těžiště jsou
            vždy číselné vektory. Posun těžiště je proto sdíleným jazykem
            obou algoritmů.

        Parameters
        ----------
        old_centroids:
            Těžiště před aktuální iterací, tvar ``(k, n_příznaků)``.
        new_centroids:
            Těžiště po aktuální iteraci, tvar ``(k, n_příznaků)``.

        Returns
        -------
        bool
            ``True`` pokud algoritmus konvergoval, jinak ``False``.
        """
        # assert: Ověřte, že obě matice těžišť mají stejný tvar
        raise NotImplementedError(
            "Úkol: implementujte _has_converged() — porovnejte posun těžišť "
            "s prahem self._EPSILON pomocí np.linalg.norm."
        )

    def fit(self, x: np.ndarray) -> IterativeClustering:
        """Natrénuje shlukování na datech iterativní optimalizací.

        Úkol:
            Implementujte iterační smyčku podle následujících kroků:

            1. Inicializujte těžiště injektovaným inicializátorem:
               ``self.centroids_ = self.initializer.initialize(x, self.k)``

            2. Opakujte až ``self.max_iter`` iterací:

               a. Přiřaďte body k těžištím (variační bod subclassy):
                  ``assignment = self._update_assignment(x, self.centroids_)``

               b. Uložte stará těžiště pro test konvergence:
                  ``old_centroids = self.centroids_.copy()``

               c. Přepočítejte těžiště z nového přiřazení (variační bod):
                  ``new_centroids = self._update_centroids(x, assignment)``

               d. Uložte výsledky a zkontrolujte konvergenci:
                  ``self.assignment_ = assignment``
                  ``self.centroids_ = new_centroids``
                  Pokud ``self._has_converged(old_centroids, new_centroids)``: break

            3. Vraťte ``self`` (umožňuje řetězení ``fit(x).predict()``).

            Pořadí operací je závazné:
            Přiřazení se počítá ze *starých* těžišť, nová těžiště z *nového*
            přiřazení. U FCM závisí matice členství na aktuálních těžištích —
            záměna pořadí by vedla k nekonzistentnímu stavu.

            Nedeterminismus je záměrný:
            Výsledek závisí na inicializaci. Opakované spuštění může dát
            různé shluky — to je přesný kontrast s deterministickým
            hierarchickým shlukováním z Cvičení 02. Není to chyba.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.

        Returns
        -------
        IterativeClustering
            Instance ``self`` po natrénování (pro řetězení metod).
        """
        # assert: Ověřte, že x je 2D matice a obsahuje alespoň k bodů
        raise NotImplementedError(
            "Úkol: implementujte fit() — iterační smyčku inicializace → přiřazení "
            "→ přepočet těžišť → konvergence. Viz docstring pro pořadí kroků."
        )

    @abstractmethod
    def _update_assignment(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Přiřadí body k těžištím (variační bod).

        K-means vrátí tvrdé popisky tvaru ``(n_bodů,)``.
        FCM vrátí matici členství tvaru ``(n_bodů, k)``.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        centroids:
            Aktuální těžiště tvaru ``(k, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Přiřazení (konkrétní tvar závisí na implementaci).
        """

    @abstractmethod
    def _update_centroids(
        self, x: np.ndarray, assignment: np.ndarray
    ) -> np.ndarray:
        """Přepočítá těžiště z aktuálního přiřazení (variační bod).

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        assignment:
            Aktuální přiřazení (tvrdé popisky nebo matice členství).

        Returns
        -------
        np.ndarray
            Nová těžiště tvaru ``(k, n_příznaků)``.
        """

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Vrátí tvrdé popisky shluků pro každý bod.

        U k-means vrátí přímo uložené popisky.
        U FCM vezme ``argmax`` matice členství.
        Vždy vrátí pole tvaru ``(n_bodů,)`` s hodnotami 0 … k-1 —
        díky tomu ``silhouette.py`` pracuje jednotně s oběma algoritmy.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``.
        """
