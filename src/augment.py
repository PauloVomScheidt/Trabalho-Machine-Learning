"""Data augmentation simples para texto no estilo EDA (Wei & Zou, 2019).

Implementa só as operações que não exigem recursos externos: troca aleatória de posição
(random swap) e remoção aleatória (random deletion) de palavras. Usado no notebook para
AVALIAR se a técnica ajuda; deve ser aplicado apenas ao conjunto de treino.
"""
from __future__ import annotations

import random

import pandas as pd


def random_swap(words: list[str], n_swaps: int, rng: random.Random) -> list[str]:
    words = words.copy()
    if len(words) < 2:
        return words
    for _ in range(n_swaps):
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def random_deletion(words: list[str], p: float, rng: random.Random) -> list[str]:
    if len(words) <= 1:
        return words
    kept = [w for w in words if rng.random() > p]
    return kept or [rng.choice(words)]


def eda_augment(text: str, alpha: float = 0.1, seed: int | None = None) -> str:
    """Gera uma variação do texto: troca ~alpha*n pares de palavras e remove cada palavra com prob. alpha.

    >>> eda_augment("o produto chegou rápido e bem embalado", seed=1) != "o produto chegou rápido e bem embalado"
    True
    """
    rng = random.Random(seed)
    words = text.split()
    n_swaps = max(1, int(alpha * len(words)))
    return " ".join(random_deletion(random_swap(words, n_swaps, rng), alpha, rng))


def augment_minority(
    train: pd.DataFrame,
    text_col: str,
    label_col: str,
    classes: list[str],
    factor: int = 1,
    alpha: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Acrescenta `factor` cópias aumentadas de cada exemplo das classes indicadas (só treino)."""
    new = []
    minority = train[train[label_col].isin(classes)]
    for k in range(factor):
        aug = minority.copy()
        aug[text_col] = [eda_augment(t, alpha, seed=seed + k * 1_000_003 + i) for i, t in enumerate(aug[text_col])]
        aug["augmented"] = True
        new.append(aug)
    base = train.assign(augmented=False)
    return pd.concat([base, *new], ignore_index=True)
