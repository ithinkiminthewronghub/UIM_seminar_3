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
    Smoke testy pro Cvičení 03 — nehierarchické shlukování.

    Testy jsou záměrně jednoduché: tři dobře oddělené syntetické shluky,
    kde správná implementace musí fungovat bez ohledu na inicializaci.

    Nedeterminismus je ošetřen:
    - k-means je testován s pevným zárodkem a tolerancí pro permutace popisků,
    - FCM je testován pouze strukturálně (součet členství = 1).

    DummyDistance: vlastní euklidovská implementace pro případ, že student
    ještě nepřepsal Distance z Cvičení 01. Odděluje testy od brány cv1.
"""

from __future__ import annotations
from itertools import permutations
import numpy as np
import pytest
from dataio.config_manager import (
    make_initializer,
    validate_config,
    ExperimentConfig,
    CommonConfig,
    DataConfig,
    KMeansConfig,
    FuzzyCMeansConfig,
)
from src.initialization import Initializer
from src.base import IterativeClustering
from src.kmeans import KMeans
from src.initialization import RandomUniformInit, ForgyInit, KMeansPlusPlusInit
from src.fuzzy_cmeans import FuzzyCMeans
from src.silhouette import silhouette_samples, silhouette_score

# ---------------------------------------------------------------------------
# Pomocná euklidovská vzdálenost — nezávislá na studentově Distance
# ---------------------------------------------------------------------------

from src.distance import Distance


class DummyDistance(Distance):
    """Jednoduchá euklidovská vzdálenost pro účely testů."""

    @property
    def is_metric(self) -> bool:
        return True

    def calculate(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        return float(np.linalg.norm(point_a - point_b))


# ---------------------------------------------------------------------------
# Syntetická data: tři dobře oddělené shluky v rovině
# ---------------------------------------------------------------------------

def make_three_clusters(random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Vygeneruje tři jasně oddělené shluky pro smoke testy.

    Returns
    -------
    X:
        Příznakový matice tvaru ``(90, 2)``.
    true_labels:
        Skutečné popisky (0, 1, 2) tvaru ``(90,)``.
    """
    rng = np.random.default_rng(random_state)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
    x_parts = [rng.normal(center, 0.3, (30, 2)) for center in centers]
    x = np.vstack(x_parts).astype(np.float32)
    true_labels = np.repeat([0, 1, 2], 30)
    return x, true_labels


def _align_labels(pred: np.ndarray, true: np.ndarray, k: int) -> np.ndarray:
    """Najde nejlepší permutaci popisků (Hungarian-lite pro malé k).

    K-means může přiřadit jiná čísla shlukům než ``true_labels`` — permutace
    je správná odpověď, ne chyba.
    """

    best_acc = -1
    best_perm = list(range(k))
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in pred])
        acc = (mapped == true).mean()
        if acc > best_acc:
            best_acc = acc
            best_perm = list(perm)
    return np.array([best_perm[p] for p in pred])


# ---------------------------------------------------------------------------
# Inicializace
# ---------------------------------------------------------------------------

class TestInitializers:
    """Testy inicializačních strategií."""

    def test_random_uniform_shape(self) -> None:
        """RandomUniformInit vrátí správný tvar těžišť."""
        x, _ = make_three_clusters()
        init = RandomUniformInit(random_state=42)
        centroids = init.initialize(x, k=3)
        assert centroids.shape == (3, 2), "Tvar těžišť musí být (k, n_příznaků)"

    def test_random_uniform_range(self) -> None:
        """Těžiště RandomUniformInit leží v rozsahu dat."""
        x, _ = make_three_clusters()
        init = RandomUniformInit(random_state=42)
        centroids = init.initialize(x, k=5)
        assert (centroids >= x.min(axis=0)).all()
        assert (centroids <= x.max(axis=0)).all()

    def test_forgy_shape(self) -> None:
        """ForgyInit vrátí správný tvar těžišť."""
        x, _ = make_three_clusters()
        init = ForgyInit(random_state=42)
        centroids = init.initialize(x, k=3)
        assert centroids.shape == (3, 2)

    def test_forgy_points_from_data(self) -> None:
        """Těžiště ForgyInit jsou existující body datasetu."""
        x, _ = make_three_clusters()
        init = ForgyInit(random_state=42)
        centroids = init.initialize(x, k=3)
        for c in centroids:
            assert any(np.allclose(c, row) for row in x), \
                "Každé těžiště Forgy musí být řádkem datasetu x"

    def test_kmeans_plus_plus_shape(self) -> None:
        """KMeansPlusPlusInit vrátí správný tvar těžišť."""
        x, _ = make_three_clusters()
        init = KMeansPlusPlusInit(random_state=42)
        centroids = init.initialize(x, k=3)
        assert centroids.shape == (3, 2)

    def test_random_state_setter_reseeds(self) -> None:
        """Setter random_state přesemení generátor — výsledky jsou reprodukovatelné."""
        x, _ = make_three_clusters()
        init = RandomUniformInit(random_state=1)
        c1 = init.initialize(x, k=3)
        init.random_state = 1
        c2 = init.initialize(x, k=3)
        np.testing.assert_array_equal(c1, c2, err_msg="Stejný seed musí dát stejný výsledek")


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------


class TestKMeans:
    """Testy k-means shlukování."""

    def test_fit_predict_shape(self) -> None:
        """predict() vrátí pole správného tvaru."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = RandomUniformInit(random_state=42)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=100)
        km.fit(x)
        labels = km.predict()
        assert labels.shape == (x.shape[0],), "predict musí vrátit (n_bodů,)"

    def test_fit_recovers_clusters(self) -> None:
        """K-means správně rozezná tři zjevně oddělené shluky."""
        x, true = make_three_clusters()
        dist = DummyDistance()
        init = ForgyInit(random_state=42)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=200)
        km.fit(x)
        pred = km.predict()
        aligned = _align_labels(pred, true, k=3)
        accuracy = (aligned == true).mean()
        assert accuracy > 0.95, f"Přesnost shlukování je příliš nízká: {accuracy:.2%}"

    def test_predict_without_fit_raises(self) -> None:
        """predict() bez předchozího fit() musí vyvolat výjimku."""
        dist = DummyDistance()
        init = RandomUniformInit(random_state=0)
        km = KMeans(k=3, distance=dist, initializer=init)
        # predict bez fit má buď NotImplementedError nebo jiný error — obě jsou OK
        with pytest.raises((NotImplementedError, RuntimeError, TypeError, AttributeError)):
            km.predict()

    def test_label_values_in_range(self) -> None:
        """Popisky musí být v rozsahu 0 … k-1."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = RandomUniformInit(random_state=7)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=100)
        km.fit(x)
        labels = km.predict()
        assert labels.min() >= 0
        assert labels.max() <= 2


# ---------------------------------------------------------------------------
# Fuzzy c-means
# ---------------------------------------------------------------------------


class TestFuzzyCMeans:
    """Testy fuzzy c-means shlukování."""

    def test_membership_rows_sum_to_one(self) -> None:
        """Každý řádek matice členství musí sumovat na 1."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = RandomUniformInit(random_state=42)
        fcm = FuzzyCMeans(k=3, distance=dist, initializer=init, q=2.0, max_iter=100)
        fcm.fit(x)
        u = fcm.assignment_
        assert u is not None, "assignment_ musí být nastaveno po fit()"
        assert u.shape == (x.shape[0], 3), f"Tvar matice členství: {u.shape}"
        np.testing.assert_allclose(
            u.sum(axis=1),
            np.ones(x.shape[0]),
            atol=1e-5,
            err_msg="Každý řádek matice členství musí sumovat na 1",
        )

    def test_predict_returns_hard_labels(self) -> None:
        """predict() FCM vrátí tvrdé popisky (argmax), ne matici."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = RandomUniformInit(random_state=42)
        fcm = FuzzyCMeans(k=3, distance=dist, initializer=init, q=2.0, max_iter=100)
        fcm.fit(x)
        labels = fcm.predict()
        assert labels.shape == (x.shape[0],), "predict FCM musí vrátit (n_bodů,)"
        assert labels.dtype in (np.int32, np.int64, np.intp), \
            "Tvrdé popisky musí být celočíselného typu"

    def test_fcm_recovers_clusters(self) -> None:
        """FCM správně rozezná tři dobře oddělené shluky."""
        x, true = make_three_clusters()
        dist = DummyDistance()
        init = ForgyInit(random_state=42)
        fcm = FuzzyCMeans(k=3, distance=dist, initializer=init, q=2.0, max_iter=200)
        fcm.fit(x)
        pred = fcm.predict()
        aligned = _align_labels(pred, true, k=3)
        accuracy = (aligned == true).mean()
        assert accuracy > 0.90, f"FCM přesnost shlukování je příliš nízká: {accuracy:.2%}"

    def test_q_parameter_stored(self) -> None:
        """Parametr q je uložen v instanci a nedostane se do základní třídy."""
        dist = DummyDistance()
        init = RandomUniformInit(random_state=0)
        fcm = FuzzyCMeans(k=2, distance=dist, initializer=init, q=3.5)
        assert fcm.q == 3.5
        assert not hasattr(fcm.__class__.__bases__[0], "q"), \
            "Parametr q nesmí být v základní třídě IterativeClustering"


# ---------------------------------------------------------------------------
# Bázová třída (Template Method)
# ---------------------------------------------------------------------------


class _ConcreteClustering(IterativeClustering):
    """Minimální konkrétní podtřída pro izolované testy bázových metod.

    Abstraktní metody implementuje triviálně — testujeme pouze sdílené
    metody ``_distances_to_centroids`` a ``_has_converged``, nezávisle
    na studentově implementaci k-means či FCM.
    """

    def _update_assignment(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0], dtype=int)

    def _update_centroids(self, x: np.ndarray, assignment: np.ndarray) -> np.ndarray:
        return self.centroids_

    def predict(self) -> np.ndarray:
        return self.assignment_


class TestBase:
    """Testy sdílených metod bázové třídy IterativeClustering."""

    @staticmethod
    def _make(k: int = 2) -> _ConcreteClustering:
        return _ConcreteClustering(
            k=k,
            distance=DummyDistance(),
            initializer=RandomUniformInit(random_state=0),
        )

    def test_distances_shape_is_rectangular(self) -> None:
        """_distances_to_centroids vrátí obdélníkovou matici (n_bodů, k)."""
        model = self._make(k=2)
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        centroids = np.array([[0.0, 0.0], [5.0, 5.0]])
        d = model._distances_to_centroids(x, centroids)
        assert d.shape == (4, 2), "Matice vzdáleností musí mít tvar (n_bodů, k)"

    def test_distances_values(self) -> None:
        """Prvky matice odpovídají euklidovské vzdálenosti bod–těžiště."""
        model = self._make(k=2)
        x = np.array([[0.0, 0.0], [3.0, 4.0]])
        centroids = np.array([[0.0, 0.0], [6.0, 8.0]])
        d = model._distances_to_centroids(x, centroids)
        expected = np.array([[0.0, 10.0], [5.0, 5.0]])
        np.testing.assert_allclose(d, expected, atol=1e-6)

    def test_has_converged_true_when_identical(self) -> None:
        """Nulový posun těžišť → konvergence (pravdivá hodnota)."""
        model = self._make(k=2)
        centroids = np.array([[0.0, 0.0], [5.0, 5.0]])
        # truthy test (ne `is True`) — implementace smí vrátit i numpy.bool_
        assert model._has_converged(centroids, centroids.copy())

    def test_has_converged_false_when_shifted(self) -> None:
        """Posun těžišť nad práh ε → nekonvergováno (nepravdivá hodnota)."""
        model = self._make(k=2)
        old = np.array([[0.0, 0.0], [5.0, 5.0]])
        new = np.array([[0.0, 0.0], [5.0, 6.0]])
        assert not model._has_converged(old, new)


# ---------------------------------------------------------------------------
# Silhoueta
# ---------------------------------------------------------------------------


class TestSilhouette:
    """Testy silhouetové analýzy."""

    def test_silhouette_samples_shape(self) -> None:
        """silhouette_samples vrátí pole správného tvaru."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = ForgyInit(random_state=42)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=100)
        km.fit(x)
        labels = km.predict()
        samples = silhouette_samples(x, labels, dist)
        assert samples.shape == (x.shape[0],)

    def test_silhouette_samples_range(self) -> None:
        """Silhouetové hodnoty leží v [-1, 1]."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = ForgyInit(random_state=42)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=100)
        km.fit(x)
        labels = km.predict()
        samples = silhouette_samples(x, labels, dist)
        assert (samples >= -1.0 - 1e-6).all() and (samples <= 1.0 + 1e-6).all(), \
            "Silhouetové hodnoty musí být v [-1, 1]"

    def test_silhouette_high_for_well_separated(self) -> None:
        """Dobře oddělené shluky mají vysoké silhouetové skóre."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = ForgyInit(random_state=42)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=200)
        km.fit(x)
        labels = km.predict()
        score = silhouette_score(x, labels, dist)
        assert score > 0.7, \
            f"Silhouetové skóre pro dobře oddělené shluky musí být > 0.7, dostali jsme {score:.3f}"

    def test_silhouette_score_is_mean(self) -> None:
        """silhouette_score je průměr silhouette_samples."""
        x, _ = make_three_clusters()
        dist = DummyDistance()
        init = RandomUniformInit(random_state=42)
        km = KMeans(k=3, distance=dist, initializer=init, max_iter=100)
        km.fit(x)
        labels = km.predict()
        samples = silhouette_samples(x, labels, dist)
        score = silhouette_score(x, labels, dist)
        np.testing.assert_allclose(score, samples.mean(), atol=1e-6)


# ---------------------------------------------------------------------------
# Továrna inicializátorů
# ---------------------------------------------------------------------------

class TestMakeInitializer:
    """Testy továrny make_initializer."""

    @pytest.mark.parametrize("name,expected_cls", [
        ("random_uniform", RandomUniformInit),
        ("forgy", ForgyInit),
        ("kmeans++", KMeansPlusPlusInit),
    ])
    def test_known_names(self, name: str, expected_cls: type) -> None:
        """Továrna vrátí správný typ inicializátoru pro každý název."""
        init = make_initializer(name, random_state=0)
        assert isinstance(init, expected_cls), \
            f"make_initializer('{name}') musí vrátit {expected_cls.__name__}"

    def test_unknown_name_raises(self) -> None:
        """Neznámý název strategie vyvolá ValueError."""
        with pytest.raises((ValueError, NotImplementedError)):
            make_initializer("neexistuje", random_state=0)

    def test_returns_initializer_instance(self) -> None:
        """Vrácená instance je podtřídou Initializer."""
        init = make_initializer("random_uniform", random_state=42)
        assert isinstance(init, Initializer)


# ---------------------------------------------------------------------------
# Validace konfigurace
# ---------------------------------------------------------------------------

def _make_cfg(
    k_km: int = 4,
    k_fcm: int = 4,
    q: float = 2.0,
    init_km: str = "kmeans++",
    init_fcm: str = "random_uniform",
) -> ExperimentConfig:
    """Sestaví validní konfiguraci, kterou jednotlivé testy záměrně rozbíjejí."""
    return ExperimentConfig(
        common=CommonConfig(random_state=42, max_iter=100),
        data=DataConfig(image="data/Bunky.png"),
        kmeans=KMeansConfig(k=k_km, initializer=init_km),
        fuzzy_cmeans=FuzzyCMeansConfig(k=k_fcm, q=q, initializer=init_fcm),
    )


class TestConfig:
    """Testy validace konfigurace (validate_config)."""

    def test_valid_config_passes(self) -> None:
        """Platná konfigurace projde bez výjimky."""
        validate_config(_make_cfg())  # nesmí nic vyhodit

    def test_k_below_two_raises(self) -> None:
        """k < 2 je neplatné pro k-means i FCM."""
        with pytest.raises((ValueError, AssertionError)):
            validate_config(_make_cfg(k_km=1))
        with pytest.raises((ValueError, AssertionError)):
            validate_config(_make_cfg(k_fcm=1))

    def test_q_not_above_one_raises(self) -> None:
        """q <= 1 způsobuje dělení nulou ve vzorci FCM → neplatné."""
        with pytest.raises((ValueError, AssertionError)):
            validate_config(_make_cfg(q=1.0))

    def test_unknown_initializer_raises(self) -> None:
        """Neznámý název inicializátoru je odmítnut."""
        with pytest.raises((ValueError, AssertionError)):
            validate_config(_make_cfg(init_km="neexistuje"))
        with pytest.raises((ValueError, AssertionError)):
            validate_config(_make_cfg(init_fcm="neexistuje"))
