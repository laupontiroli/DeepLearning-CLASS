"""Geração de dados e perceptron artesanal para a atividade 2. Perceptron.

Este módulo contém somente utilitários NumPy. O fluxo principal, a ordem das
amostragens e as figuras ficam no notebook perceptron_activity.ipynb.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class TrainingResult:
    """Pesos finais, histórico de treino e, opcionalmente, o snapshot pocket."""

    weights: FloatArray
    bias: float
    epochs: int
    accuracy_history: list[float]
    updates_per_epoch: list[int]
    pocket_weights: Optional[FloatArray] = None
    pocket_bias: Optional[float] = None
    pocket_accuracy: Optional[float] = None
    pocket_epoch: Optional[int] = None
    pocket_accuracy_history: Optional[list[float]] = None


def make_data(
    rng: np.random.Generator,
    mean_class_0: list[float],
    mean_class_1: list[float],
    variance: float,
    samples_per_class: int = 1000,
) -> tuple[FloatArray, IntArray]:
    """Gera as duas classes gaussianas, sem embaralhar: classe 0 vem primeiro."""
    covariance = variance * np.eye(2)
    class_0 = rng.multivariate_normal(
        mean_class_0, covariance, size=samples_per_class
    )
    class_1 = rng.multivariate_normal(
        mean_class_1, covariance, size=samples_per_class
    )
    features = np.vstack((class_0, class_1))
    labels = np.concatenate(
        (
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        )
    )
    return features, labels


def predict(features: FloatArray, weights: FloatArray, bias: float) -> IntArray:
    """Aplica step(z): 1 quando w·x + b >= 0; 0 nos demais casos."""
    return (features @ weights + bias >= 0).astype(np.int64)


def accuracy(features: FloatArray, labels: IntArray, weights: FloatArray, bias: float) -> float:
    """Calcula a fração de previsões corretas no conjunto completo."""
    return float(np.mean(predict(features, weights, bias) == labels))


def train_perceptron(
    features: FloatArray,
    labels: IntArray,
    initial_weights: FloatArray,
    initial_bias: float = 0.0,
    learning_rate: float = 0.01,
    max_epochs: int = 100,
    use_pocket: bool = False,
) -> TrainingResult:
    """Treina em ordem fixa com rótulos 0/1 e, se pedido, acompanha o pocket.

    Para cada ponto, o erro é y - y_hat. Apenas erros não nulos alteram os
    parâmetros. O pocket é avaliado sobre todos os dados após cada atualização.
    """
    weights = initial_weights.astype(np.float64, copy=True)
    bias = float(initial_bias)
    accuracy_history: list[float] = []
    updates_per_epoch: list[int] = []

    # O estado inicial também é candidato ao pocket; a comparação posterior é estrita.
    best_weights = weights.copy()
    best_bias = bias
    best_accuracy = accuracy(features, labels, weights, bias)
    best_epoch = 0
    pocket_accuracy_history: list[float] = []

    for epoch in range(1, max_epochs + 1):
        updates = 0
        for feature, label in zip(features, labels):
            # A regra usa explicitamente y - y_hat para rótulos binários 0/1.
            prediction = int(np.dot(weights, feature) + bias >= 0)
            error = int(label) - prediction
            if error != 0:
                weights += learning_rate * error * feature
                bias += learning_rate * error
                updates += 1

                # A única extensão pocket: medir o conjunto todo após cada atualização.
                if use_pocket:
                    current_accuracy = accuracy(features, labels, weights, bias)
                    if current_accuracy > best_accuracy:
                        best_weights = weights.copy()
                        best_bias = bias
                        best_accuracy = current_accuracy
                        best_epoch = epoch

        accuracy_history.append(accuracy(features, labels, weights, bias))
        updates_per_epoch.append(updates)
        if use_pocket:
            pocket_accuracy_history.append(best_accuracy)

        # Só encerra antes do limite quando uma época inteira não atualiza nenhum peso.
        if updates == 0:
            break

    if use_pocket:
        return TrainingResult(
            weights=weights,
            bias=bias,
            epochs=len(accuracy_history),
            accuracy_history=accuracy_history,
            updates_per_epoch=updates_per_epoch,
            pocket_weights=best_weights,
            pocket_bias=best_bias,
            pocket_accuracy=best_accuracy,
            pocket_epoch=best_epoch,
            pocket_accuracy_history=pocket_accuracy_history,
        )

    return TrainingResult(
        weights=weights,
        bias=bias,
        epochs=len(accuracy_history),
        accuracy_history=accuracy_history,
        updates_per_epoch=updates_per_epoch,
    )
