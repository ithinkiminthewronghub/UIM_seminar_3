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
    Cvičení 03 Umělá inteligence v medicíně — Nehierarchické shlukování
"""

from __future__ import annotations

import sys


def _ni(name: str, exc: NotImplementedError) -> None:
    """Vypíše přátelskou zprávu o nedokončeném úkolu a pokračuje."""
    print(f"\n[NEDOKONCENO] {name}")
    print(f"  > {exc}")
    print("  Implementujte chybejici cast a spustte znovu.\n")


# ---------------------------------------------------------------------------
# Načtení konfigurace
# ---------------------------------------------------------------------------
try:
    from dataio import (
        load_config,
        make_initializer,
        load_image,
        to_hsv,
        select_channels,
        plot_segmentation,
        plot_silhouette,
        plot_k_selection,
    )

    from src import (
        EuclideanDistance,
        KMeans,
        FuzzyCMeans,
        silhouette_samples,
        silhouette_score
    )

except ImportError as e:
    print(f"Chyba importu: {e}")
    sys.exit(1)

print("=" * 60)
print("Cvičení 03 — Nehierarchické shlukování")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Konfigurace
# ---------------------------------------------------------------------------
# load_config volá validate_config — dokud není ověření implementováno,
# nelze načíst konfiguraci a pipeline se korektně ukončí.
try:
    cfg = load_config("config.yaml")
except NotImplementedError as e:
    _ni("load_config / validate_config", e)
    sys.exit(0)
except (ValueError, AssertionError) as e:
    print(f"Neplatná konfigurace: {e}")
    sys.exit(1)

common = cfg.common
km_cfg = cfg.kmeans
fcm_cfg = cfg.fuzzy_cmeans

print(f"\nKonfigurace načtena: k-means k={km_cfg.k}, FCM k={fcm_cfg.k}, q={fcm_cfg.q}")

# ---------------------------------------------------------------------------
# 2. Načtení dat
# ---------------------------------------------------------------------------
# Cesta k obrázku je v config.yaml (sekce data) — přepnutí na jiný snímek
# je otázkou úpravy konfigurace, ne kódu.
try:
    rgb_data, dimensions = load_image(cfg.data.image)
    print(f"Obrázek načten: {dimensions[0]}×{dimensions[1]} pixelů, "
          f"příznakový matice {rgb_data.shape}")
except (OSError, ValueError) as e:
    print(f"Chyba pri nacitani obrazku: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Sdílené závislosti
# ---------------------------------------------------------------------------
dist = EuclideanDistance()

# ---------------------------------------------------------------------------
# 4. K-means
# ---------------------------------------------------------------------------
print("\n--- K-means ---")
try:
    km_initializer = make_initializer(km_cfg.initializer, common.random_state)
    km = KMeans(
        k=km_cfg.k,
        distance=dist,
        initializer=km_initializer,
        max_iter=common.max_iter,
    )
    km.fit(rgb_data)
    km_labels = km.predict()
    print(f"K-means dokončen. Počet shluků: {km_cfg.k}")
    plot_segmentation(km_labels, dimensions, km_cfg.k, title=f"K-means (k={km_cfg.k})")
except NotImplementedError as e:
    _ni("K-means fit/predict", e)
    km_labels = None

# Silhoueta pro k-means
if km_labels is not None:
    print("Silhoueta pro k-means...")
    try:
        km_sil = silhouette_samples(rgb_data, km_labels, dist)
        print(f"  Průměrné silhouetové skóre: {km_sil.mean():.4f}")
        plot_silhouette(km_sil, km_labels, title="Silhoueta — K-means")
    except NotImplementedError as e:
        _ni("silhouette_samples (k-means)", e)

# ---------------------------------------------------------------------------
# 5. Fuzzy c-means
# ---------------------------------------------------------------------------
print("\n--- Fuzzy c-means ---")
try:
    fcm_initializer = make_initializer(fcm_cfg.initializer, common.random_state)
    fcm = FuzzyCMeans(
        k=fcm_cfg.k,
        distance=dist,
        initializer=fcm_initializer,
        q=fcm_cfg.q,
        max_iter=common.max_iter,
    )
    fcm.fit(rgb_data)
    fcm_labels = fcm.predict()
    print(f"FCM dokončen. Počet shluků: {fcm_cfg.k}, q={fcm_cfg.q}")
    plot_segmentation(fcm_labels, dimensions, fcm_cfg.k,
                      title=f"Fuzzy c-means (k={fcm_cfg.k}, q={fcm_cfg.q})")
except NotImplementedError as e:
    _ni("FuzzyCMeans fit/predict", e)
    fcm_labels = None

# Silhoueta pro FCM
if fcm_labels is not None:
    print("Silhoueta pro FCM...")
    try:
        fcm_sil = silhouette_samples(rgb_data, fcm_labels, dist)
        print(f"  Průměrné silhouetové skóre: {fcm_sil.mean():.4f}")
        plot_silhouette(
            fcm_sil,
            fcm_labels,
            title="Silhoueta — Fuzzy c-means",
            save=True,
            )
    except NotImplementedError as e:
        _ni("silhouette_samples (FCM)", e)

# ---------------------------------------------------------------------------
# 6. Experiment: vliv inicializace (Cíl 4)
# ---------------------------------------------------------------------------
print("\n--- Experiment: srovnání inicializačních strategií ---")
for init_name in ["random_uniform", "forgy", "kmeans++"]:
    try:
        init = make_initializer(init_name, common.random_state)
        model = KMeans(
            k=km_cfg.k,
            distance=dist,
            initializer=init,
            max_iter=common.max_iter,
        )
        model.fit(rgb_data)
        labels = model.predict()
        try:
            score = silhouette_score(rgb_data, labels, dist)
            print(f"  {init_name:20s}: silhouetové skóre = {score:.4f}")
        except NotImplementedError:
            print(f"  {init_name:20s}: shlukování OK, silhoueta nedokončena")
    except NotImplementedError as e:
        _ni(f"Inicializace '{init_name}'", e)

# ---------------------------------------------------------------------------
# 7. Experiment: různé barevné kanály
# ---------------------------------------------------------------------------
print("\n--- Experiment: výběr barevných kanálů ---")
channel_configs = {
    "R+G": [0, 1],
    "R+B": [0, 2],
    "G+B": [1, 2],
}
for ch_name, channels in channel_configs.items():
    try:
        subset = select_channels(rgb_data, channels)
        init = make_initializer("random_uniform", common.random_state)
        model = KMeans(
            k=km_cfg.k,
            distance=dist,
            initializer=init,
            max_iter=common.max_iter,
        )
        model.fit(subset)
        labels = model.predict()
        try:
            score = silhouette_score(subset, labels, dist)
            print(f"  Kanaly {ch_name}: silhouetove skore = {score:.4f}")
        except NotImplementedError:
            print(f"  Kanaly {ch_name}: shlukování OK, silhoueta nedokoncena")
    except NotImplementedError as e:
        _ni(f"Kanaly {ch_name}", e)

# ---------------------------------------------------------------------------
# 8. Experiment: prostor HSV
# ---------------------------------------------------------------------------
print("\n--- Experiment: HSV prostor ---")
try:
    hsv_data = to_hsv(rgb_data)
    init = make_initializer("random_uniform", common.random_state)
    model = KMeans(
        k=km_cfg.k,
        distance=dist,
        initializer=init,
        max_iter=common.max_iter,
    )
    model.fit(hsv_data)
    hsv_labels = model.predict()
    plot_segmentation(
        hsv_labels,
        dimensions,
        km_cfg.k,
        title="K-means v prostoru HSV",
        save=True,
        )
    try:
        score = silhouette_score(hsv_data, hsv_labels, dist)
        print(f"  HSV prostor: silhouetové skóre = {score:.4f}")
    except NotImplementedError as e:
        _ni("silhouette_score (HSV)", e)
except NotImplementedError as e:
    _ni("to_hsv", e)

# ---------------------------------------------------------------------------
# 9. Výběr optimálního k (silhouetová křivka)
# ---------------------------------------------------------------------------
print("\n--- Výběr optimálního počtu shluků k ---")
k_range = list(range(2, 8))
k_scores: list[float] = []
k_valid: list[int] = []

for k_val in k_range:
    try:
        init = make_initializer("random_uniform", common.random_state)
        model = KMeans(
            k=k_val,
            distance=dist,
            initializer=init,
            max_iter=common.max_iter,
        )
        model.fit(rgb_data)
        labels = model.predict()
        score = silhouette_score(rgb_data, labels, dist)
        k_scores.append(score)
        k_valid.append(k_val)
        print(f"  k={k_val}: skóre = {score:.4f}")
    except NotImplementedError as e:
        _ni(f"Výběr k (k={k_val})", e)
        break

if k_valid:
    plot_k_selection(
        k_valid,
        k_scores,
        save=True,
        title="Výběr optimálního počtu shluků k",
        )

print("\n" + "=" * 60)
print("Pipeline dokončen.")
print("=" * 60)
