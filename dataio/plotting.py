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
    Vizualizační funkce pro výsledky nehierarchického shlukování.

    Tři tenké obálky — vykreslení je předpřipraveno, studenti se soustředí
    na samotný algoritmus a interpretaci výsledků.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_GRAPHS_DIR = Path("graphs")


def _save_figure(title: str) -> None:
    """Uloží aktuální figuru do složky graphs/ jako PDF.

    Název souboru je odvozen z nadpisu: diakritika se odstraní, mezery
    a speciální znaky se nahradí podtržítkem.
    """
    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = normalized.encode("ascii", "ignore").decode("ascii")
    safe_name = re.sub(r"[^\w]+", "_", ascii_title).strip("_").lower()
    filepath = _GRAPHS_DIR / f"{safe_name}.pdf"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")


def plot_segmentation(
    labels: np.ndarray,
    dimensions: tuple[int, int],
    k: int,
    title: str = "Segmentace",
    save: bool = False,
) -> None:
    """Zobrazí segmentovaný obrázek obarvený podle přiřazených shluků.

    Plochý vektor popisků je přetvarován zpět na mřížku pixelů a zobrazen
    jako šedotónový obrázek (hodnoty normalizovány do [0, 1]).

    Parameters
    ----------
    labels:
        Pole tvaru ``(n_pixelů,)`` s čísly shluků 0 … k-1.
    dimensions:
        Dvojice ``(výška, šířka)`` pro přetvarování — vrácena z ``load_image``.
    k:
        Počet shluků (použit pro normalizaci barev).
    title:
        Nadpis obrázku.
    save:
        Pokud ``True``, uloží figuru do složky ``graphs/`` jako PDF.
    """
    height, width = dimensions
    segmented: np.ndarray = labels.reshape(height, width)

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(segmented, cmap="tab10", vmin=0, vmax=max(k - 1, 1))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save:
        _save_figure(title)
    plt.show(
        block=False
    )


def plot_silhouette(
    sample_values: np.ndarray,
    labels: np.ndarray,
    title: str = "Silhouetový diagram",
    save: bool = False,
) -> None:
    """Klasický silhouetový diagram tříděný podle shluků.

    Pro každý shluk jsou zobrazeny silhouetové hodnoty jeho pixelů jako
    horizontální pruhy seřazené sestupně. Červená svislá čára označuje
    průměrnou hodnotu přes všechny body.

    Parameters
    ----------
    sample_values:
        Pole tvaru ``(n_pixelů,)`` — výstup ``silhouette_samples``.
    labels:
        Pole tvaru ``(n_pixelů,)`` s přiřazením do shluků.
    title:
        Nadpis diagramu.
    save:
        Pokud ``True``, uloží figuru do složky ``graphs/`` jako PDF.
    """
    k: int = int(labels.max()) + 1
    _, ax = plt.subplots(figsize=(8, 5))
    y_lower: int = 0

    colors = plt.colormaps["tab10"](np.linspace(0, 1, k))

    for cluster_idx in range(k):
        mask: np.ndarray = labels == cluster_idx
        cluster_values: np.ndarray = np.sort(sample_values[mask])[::-1]
        cluster_size: int = cluster_values.shape[0]
        y_upper: int = y_lower + cluster_size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            facecolor=colors[cluster_idx],
            alpha=0.8,
            label=f"Shluk {cluster_idx}",
        )
        y_lower = y_upper + 5  # malá mezera mezi shluky

    mean_val: float = float(sample_values.mean())
    ax.axvline(x=mean_val, color="red", linestyle="--", label=f"Průměr: {mean_val:.3f}")
    ax.set_xlabel("Silhouetová hodnota")
    ax.set_ylabel("Pixely (seřazeny v rámci shluku)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save:
        _save_figure(title)
    plt.show(
        block=False
    )


def plot_k_selection(
    k_values: list[int],
    scores: list[float],
    title: str = "Výběr počtu shluků k",
    save: bool = False,
) -> None:
    """Zobrazí průměrné silhouetové skóre v závislosti na počtu shluků k.

    Maximum je zvýrazněno červenou hvězdičkou — pomáhá studentům
    identifikovat optimální počet shluků.

    Parameters
    ----------
    k_values:
        Seznam testovaných hodnot k.
    scores:
        Průměrné silhouetové skóre pro každé k (stejná délka jako ``k_values``).
    title:
        Nadpis grafu.
    save:
        Pokud ``True``, uloží figuru do složky ``graphs/`` jako PDF.
    """
    best_idx: int = int(np.argmax(scores))

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, scores, marker="o", linewidth=2, label="Průměrné silhouetové skóre")
    ax.scatter(
        [k_values[best_idx]],
        [scores[best_idx]],
        color="red",
        zorder=5,
        s=150,
        marker="*",
        label=f"Optimální k = {k_values[best_idx]} (skóre = {scores[best_idx]:.3f})",
    )
    ax.set_xlabel("Počet shluků k")
    ax.set_ylabel("Průměrné silhouetové skóre")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save:
        _save_figure(title)
    plt.show(
        block=False
    )
