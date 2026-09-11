"""Exercise 1 — Exploring class separability in 2D at multiple spreads.

Gera quatro conjuntos gaussianos equivalentes, com os desvios multiplicados por cada
``scale``. Salva as figuras individuais, a comparação em subplots e a curva de taxa de
mistura, além de imprimir as métricas que alimentam o relatório.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_multiple_spreads.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
S = [0.5, 1.0, 2.0, 4.0]
RNG_SEED = 42

CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}
N_PER_CLASS = 100
MEANS = np.array([CLASSES[c]["mean"] for c in CLASSES], dtype=float)


def build_base_noise() -> dict[int, np.ndarray]:
    """Gera ruído base determinístico para reutilizar nos diferentes scales."""
    rng = np.random.default_rng(RNG_SEED)
    return {
        label: rng.normal(0.0, 1.0, size=(N_PER_CLASS, 2))
        for label in CLASSES
    }


def generate(scale: float, base_noise: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Amostra 100 pontos por classe, com os desvios multiplicados por ``scale``."""
    xs, ys = [], []
    for label, params in CLASSES.items():
        mean = np.asarray(params["mean"])
        std = np.asarray(params["std"])
        xs.append(mean + base_noise[label] * std * scale)
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def pairwise_separation_ratios(X: np.ndarray, y: np.ndarray) -> dict[tuple[int, int], float]:
    """Mede a separação entre cada par de classes na escala atual."""
    centroids = np.stack([X[y == c].mean(axis=0) for c in CLASSES])
    spreads = np.array(
        [np.linalg.norm(X[y == c] - centroids[c], axis=1).mean() for c in CLASSES]
    )
    ratios: dict[tuple[int, int], float] = {}
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
        pair_spread = (spreads[i] + spreads[j]) / 2.0
        ratios[(i, j)] = float(centroid_distance / pair_spread)
    return ratios


def mixing_rate(X: np.ndarray, y: np.ndarray) -> float:
    """Fração de pontos cujo centro mais próximo não é o da própria classe."""
    distances = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    nearest_mean = distances.argmin(axis=1)
    return float(np.mean(nearest_mean != y))


def plot_figure_2(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    """Salva a Figura 2 com quatro subplots compartilhando os mesmos limites."""
    all_X = np.vstack([data[0] for data in datasets.values()])
    x_min, x_max = all_X[:, 0].min(), all_X[:, 0].max()
    y_min, y_max = all_X[:, 1].min(), all_X[:, 1].max()
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, scale in zip(axes, S):
        X, y = datasets[scale]
        centroids = np.stack([X[y == c].mean(axis=0) for c in CLASSES])
        for c in CLASSES:
            ax.scatter(*X[y == c].T, s=14, alpha=0.75, label=f"Classe {c}")
        ax.scatter(centroids[:, 0], centroids[:, 1], s=80, marker="X", color="black", label="Centro")
        ax.set_title(f"scale = {scale}")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal", adjustable="box")

    axes[0].legend(loc="upper left")
    fig.supxlabel("$x_1$")
    fig.supylabel("$x_2$")
    fig.suptitle("Figura 2 — Comparação das nuvens gaussianas para diferentes scales")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES / "fig02-point-clouds.png", dpi=150)
    plt.close(fig)


def plot_figure_3(scales: list[float], rates: list[float]) -> None:
    """Salva a Figura 3 com a taxa de mistura em função do scale."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scales, rates, marker="o", linewidth=2.0)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Mixing rate")
    ax.set_title("Figura 3 — Taxa de mistura em função do scale")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    base_noise = build_base_noise()
    datasets = {scale: generate(scale, base_noise) for scale in S}

    plot_figure_2(datasets)

    rates = {scale: mixing_rate(X, y) for scale, (X, y) in datasets.items()}
    plot_figure_3(S, [rates[scale] for scale in S])

    scale_1 = 1.0
    scale_1_ratios = pairwise_separation_ratios(*datasets[scale_1])
    smallest_pair, smallest_ratio = min(scale_1_ratios.items(), key=lambda item: item[1])

    print("Pairwise separation ratios (scale = 1.0):")
    for (i, j), ratio in scale_1_ratios.items():
        print(f"  classes ({i}, {j}) -> {ratio:.3f}")
    print(f"\nSmallest pair ratio at scale=1.0: classes {smallest_pair} = {smallest_ratio:.3f}")

    print("\nMixing rates:")
    for scale in S:
        print(f"  scale={scale:>4} | mixing rate = {rates[scale]:.3f}")

    smallest_per_scale = {}
    for scale in S:
        ratios = pairwise_separation_ratios(*datasets[scale])
        smallest_per_scale[scale] = min(ratios.values())

    print("\nSmallest pairwise ratio by scale:")
    for scale in S:
        print(f"  scale={scale:>4} | smallest pair ratio = {smallest_per_scale[scale]:.3f}")

    threshold_scale = next(
        (scale for scale in S if smallest_per_scale[scale] < 1.0),
        None,
    )
    if threshold_scale is not None:
        print(
            f"\nThe smallest pairwise ratio drops below 1.0 first at scale={threshold_scale}. "
            f"Its value is {smallest_per_scale[threshold_scale]:.3f}."
        )
    else:
        print("\nThe smallest pairwise ratio never drops below 1.0 in the tested scales.")


if __name__ == "__main__":
    main()
