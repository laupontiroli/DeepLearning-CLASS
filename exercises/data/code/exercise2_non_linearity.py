"""Exercise 2 — Non-linearity in higher dimensions.

Gera os dois datasets sintéticos em 5D, reduz para 2D com PCA e salva as figuras
solicitadas, além de imprimir as métricas para o relatório.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)

MU_A = np.zeros(5)
SIGMA_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ],
    dtype=float,
)

MU_B = np.ones(5) * 1.5
SIGMA_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ],
    dtype=float,
)


def make_dataset_i() -> tuple[np.ndarray, np.ndarray]:
    A = RNG.multivariate_normal(MU_A, SIGMA_A, size=500)
    B = RNG.multivariate_normal(MU_B, SIGMA_B, size=500)
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    return X, y


def make_dataset_ii() -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((1000, 5), dtype=float)
    y = np.empty(1000, dtype=int)

    dirs = RNG.normal(size=(500, 5))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = 1.5 + 0.05 * RNG.normal(size=500)
    X[:500] = dirs * radii[:, None]
    y[:500] = 0

    dirs = RNG.normal(size=(500, 5))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = 5.0 + 0.05 * RNG.normal(size=500)
    X[500:] = dirs * radii[:, None]
    y[500:] = 1

    return X, y


def pca_projection(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    return X2, pca.explained_variance_ratio_


def figure_4(X1: np.ndarray, y1: np.ndarray, X2: np.ndarray, y2: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X1[y1 == 0, 0], X1[y1 == 0, 1], alpha=0.6, label="Class A")
    axes[0].scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], alpha=0.6, label="Class B")
    axes[0].legend()
    axes[0].set_title("Dataset I — PCA 5D → 2D")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], alpha=0.6, label="Class C")
    axes[1].scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], alpha=0.6, label="Class D")
    axes[1].legend()
    axes[1].set_title("Dataset II — PCA 5D → 2D")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-pca-comparison.png", dpi=150)
    plt.close(fig)


def figure_5(X1: np.ndarray, y1: np.ndarray, X2: np.ndarray, y2: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    radii1 = np.linalg.norm(X1, axis=1)
    radii2 = np.linalg.norm(X2, axis=1)

    for ax, radii, label, y in [
        (axes[0], radii1, "Dataset I", y1),
        (axes[1], radii2, "Dataset II", y2),
    ]:
        ax.hist(radii[y == 0], bins=20, alpha=0.6, label="Classe A/C", density=True)
        ax.hist(radii[y == 1], bins=20, alpha=0.6, label="Classe B/D", density=True)
        ax.set_title(f"{label} — histograma do raio")
        ax.set_xlabel("||x||_2")
        ax.set_ylabel("densidade")
        ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-radius-histograms.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    X1, y1 = make_dataset_i()
    X2, y2 = make_dataset_ii()

    X1_pca, ev1 = pca_projection(X1)
    X2_pca, ev2 = pca_projection(X2)

    figure_4(X1_pca, y1, X2_pca, y2)
    figure_5(X1, y1, X2, y2)

    center_distance_i = np.linalg.norm(X1[y1 == 0].mean(axis=0) - X1[y1 == 1].mean(axis=0))
    center_distance_ii = np.linalg.norm(X2[y2 == 0].mean(axis=0) - X2[y2 == 1].mean(axis=0))

    print("Dataset I:")
    print(f"  center distance = {center_distance_i:.3f}")
    print(f"  explained variance PC1 + PC2 = {ev1.sum():.4f}")

    print("\nDataset II:")
    print(f"  center distance = {center_distance_ii:.3f}")
    print(f"  explained variance PC1 + PC2 = {ev2.sum():.4f}")

    print("\nRadius separation cues:")
    radii1 = np.linalg.norm(X1, axis=1)
    radii2 = np.linalg.norm(X2, axis=1)
    print(f"  Dataset I mean radius class A = {radii1[y1 == 0].mean():.3f}; class B = {radii1[y1 == 1].mean():.3f}")
    print(f"  Dataset II mean radius class C = {radii2[y2 == 0].mean():.3f}; class D = {radii2[y2 == 1].mean():.3f}")


if __name__ == "__main__":
    main()
