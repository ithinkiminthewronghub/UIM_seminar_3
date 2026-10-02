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
    Silhouetová analýza kvality shlukování.

    Měří, jak dobře každý bod zapadá do svého shluku v porovnání s nejbližším
    sousedním shlukem. Pracuje s tvrdými popisky — kompatibilní s k-means i FCM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.distance import Distance


def silhouette_samples(
    x: np.ndarray,
    labels: np.ndarray,
    distance: "Distance",
) -> np.ndarray:
    """Vypočítá silhouetovou hodnotu pro každý bod datasetu.

    Úkol:
        Implementujte výpočet silhouetové hodnoty ``s(i)`` pro každý bod ``i``:

        .. math::
            s(i) = \\frac{b(i) - a(i)}{\\max(a(i),\\, b(i))}

        kde:

        - ``a(i)`` = průměrná vzdálenost bodu ``i`` od všech ostatních bodů
          ve **stejném** shluku (míra soudržnosti),
        - ``b(i)`` = průměrná vzdálenost bodu ``i`` od všech bodů v **nejbližším
          jiném** shluku (míra oddělenosti).

        Postup:
        1. Pro každý bod ``i`` identifikujte jeho shluk ``c = labels[i]``.
        2. Spočítejte ``a(i)`` jako průměr vzdáleností ke všem ostatním bodům
           se stejným popiskem (vylučte bod ``i`` samotný).
        3. Pro každý jiný shluk ``c'`` spočítejte průměrnou vzdálenost bodu
           ``i`` od všech bodů shluku ``c'``. Hodnota ``b(i)`` je minimum
           těchto průměrů přes všechny ``c' ≠ c``.
        4. Vraťte ``(b(i) - a(i)) / max(a(i), b(i))``.

        Speciální případ: pokud shluk obsahuje jen jeden bod, nastavte ``s(i) = 0``.

        Pro výpočet vzdáleností volejte ``distance.calculate(X[i], X[j])`` —
        zajišťuje konzistenci s metrikou použitou při shlukování.

    Parameters
    ----------
    x:
        Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
    labels:
        Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.
    distance:
        Instance metriky vzdálenosti — stejná jako při shlukování.

    Returns
    -------
    np.ndarray
        Silhouetové hodnoty, tvar ``(n_bodů,)``, hodnoty v [-1, 1].
        Vyšší hodnota znamená lepší zařazení bodu do shluku.
    """
    # assert: Ověřte, že x je 2D matice, labels je 1D pole stejné délky
    # a obsahuje alespoň 2 různé shluky
    assert x.ndim == 2
    assert labels.ndim == 1
    assert x.shape[0] == labels.shape[0]

    unique_clusters = np.unique(labels)
    assert len(unique_clusters) >= 2

    n_samples = x.shape[0]
    sil_samples = np.zeros(n_samples, dtype=np.float64)

    # Předpočítání vzdáleností mezi všemi dvojicemi bodů
    # pro úsporu opakovaných volání distance.calculate
    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = distance.calculate(x[i], x[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    for i in range(n_samples):
        c_i = labels[i]

        # 1. Výpočet a(i): průměrná vzdálenost k ostatním bodům ve stejném shluku
        same_cluster_indices = np.where((labels == c_i) & (np.arange(n_samples) != i))[0]

        # Speciální případ: pokud shluk obsahuje pouze jeden bod, s(i) = 0
        if len(same_cluster_indices) == 0:
            sil_samples[i] = 0.0
            continue

        a_i = np.mean(dist_matrix[i, same_cluster_indices])

        # 2. Výpočet b(i): průměrná vzdálenost k bodům v nejbližším jiném shluku
        b_i = np.inf
        for c in unique_clusters:
            if c == c_i:
                continue

            other_cluster_indices = np.where(labels == c)[0]
            if len(other_cluster_indices) > 0:
                mean_dist_to_c = np.mean(dist_matrix[i, other_cluster_indices])
                if mean_dist_to_c < b_i:
                    b_i = mean_dist_to_c

        # 3. Výpočet s(i) = (b(i) - a(i)) / max(a(i), b(i))
        max_ab = max(a_i, b_i)
        if max_ab == 0:
            sil_samples[i] = 0.0
        else:
            sil_samples[i] = (b_i - a_i) / max_ab

    return sil_samples


def silhouette_score(
    x: np.ndarray,
    labels: np.ndarray,
    distance: "Distance",
) -> float:
    """Vypočítá průměrné silhouetové skóre přes všechny body.

    Úkol:
        Zavolejte ``silhouette_samples`` a vraťte průměr výsledného pole.
        Průměrné skóre slouží jako jednočíselná míra kvality shlukování —
        používá se pro výběr optimálního ``k``.

    Parameters
    ----------
    x:
        Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
    labels:
        Tvrdé popisky shluků, tvar ``(n_bodů,)``.
    distance:
        Instance metriky vzdálenosti.

    Returns
    -------
    float
        Průměrné silhouetové skóre v rozsahu [-1, 1].
        Blíže k 1 → kvalitnější shlukování.
    """
    samples = silhouette_samples(x, labels, distance)
    return float(np.mean(samples))
