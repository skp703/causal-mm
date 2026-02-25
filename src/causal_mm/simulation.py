from __future__ import annotations

import numpy as np

from .fcm import FCMGraph


def adjacency_from_estimates(fcm: FCMGraph) -> np.ndarray:
    """
    Return adjacency matrix using scaled_weight for each edge.
    """

    return fcm.adjacency_matrix(use_estimates=True)


def simulate_fcm(
    fcm: FCMGraph,
    initial_state: np.ndarray,
    n_steps: int = 10,
    activation: str = "tanh",
) -> np.ndarray:
    """
    Simulate FCM dynamics: x^{t+1} = f(A x^t)
    """

    A = adjacency_from_estimates(fcm)
    x = np.array(initial_state, dtype=float).reshape(-1, 1)
    history = [x.flatten()]

    def _act(z):
        if activation == "tanh":
            return np.tanh(z)
        if activation == "logistic":
            return 1 / (1 + np.exp(-z))
        if activation == "identity":
            return z
        raise ValueError(f"Unsupported activation: {activation}")

    for _ in range(n_steps):
        x = _act(A @ x)
        history.append(x.flatten())
    return np.vstack(history)
