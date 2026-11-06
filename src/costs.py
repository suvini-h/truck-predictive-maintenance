"""Cost matrix, planner actions and cost helpers shared by notebooks 02-04.

COST[actual][predicted] is the IDA 2024 challenge matrix (relative units set by Scania
experts, not SEK). Headline results are cost per 1,000 trucks.
"""

from __future__ import annotations

import numpy as np

COST = np.array([
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
])
CLASSES = [0, 1, 2, 3, 4]
ACTIONS = {
    4: "Call in now",
    3: "Book within 12 steps",
    2: "Book within 24 steps",
    1: "Plan within 48 steps",
    0: "No action",
}


def total_cost(actual, predicted) -> float:
    return float(COST[np.asarray(actual, dtype=int), np.asarray(predicted, dtype=int)].sum())


def cost_per_1000(actual, predicted) -> float:
    actual = np.asarray(actual)
    return 1000 * total_cost(actual, predicted) / len(actual)


def expected_cost(proba) -> np.ndarray:
    """(n, 5) class probabilities -> (n, 5) expected cost of each predicted action."""
    return np.asarray(proba) @ COST


def expected_cost_decision(proba) -> np.ndarray:
    """Bayes decision rule: the action with the lowest expected cost."""
    return expected_cost(proba).argmin(axis=1)
