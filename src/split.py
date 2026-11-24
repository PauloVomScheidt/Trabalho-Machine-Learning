"""Divisão treino/validação/teste e artefatos que dependem só do treino (pesos de classe)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "submission_date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide por ordem cronológica: mais antigos → treino, seguintes → validação, recentes → teste.

    Os cortes são feitos por posição na ordem temporal (não por mês inteiro), então as
    proporções ficam próximas de 70/15/15 mesmo com meses de volumes diferentes.

    >>> d = pd.DataFrame({"submission_date": pd.date_range("2018-01-01", periods=20, freq="D")})
    >>> [len(p) for p in temporal_split(d)]
    [14, 3, 3]
    """
    ordered = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    n = len(ordered)
    i_train = int(round(n * train_frac))
    i_val = int(round(n * (train_frac + val_frac)))
    return ordered.iloc[:i_train], ordered.iloc[i_train:i_val], ordered.iloc[i_val:]


def stratified_split(
    df: pd.DataFrame, label_col: str, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Alternativa aleatória estratificada (usada só se a divisão temporal não for viável)."""
    train, rest = train_test_split(df, train_size=train_frac, stratify=df[label_col], random_state=seed)
    val_share = val_frac / (1 - train_frac)
    val, test = train_test_split(rest, train_size=val_share, stratify=rest[label_col], random_state=seed)
    return train, val, test


def split_summary(parts: dict[str, pd.DataFrame], label_col: str, date_col: str) -> pd.DataFrame:
    """Tamanho, período e distribuição de classes (%) de cada conjunto."""
    rows = []
    total = sum(len(p) for p in parts.values())
    for name, p in parts.items():
        dist = p[label_col].value_counts(normalize=True).mul(100).round(1)
        rows.append(
            {
                "conjunto": name,
                "linhas": len(p),
                "% do total": round(100 * len(p) / total, 1),
                "início": p[date_col].min(),
                "fim": p[date_col].max(),
                **{f"% {k}": v for k, v in dist.sort_index().items()},
            }
        )
    return pd.DataFrame(rows).set_index("conjunto")


def class_weights_from_train(y_train: np.ndarray, classes: np.ndarray) -> dict[int, float]:
    """Pesos 'balanced' (n_amostras / (n_classes * n_amostras_da_classe)) calculados SÓ no treino.

    >>> class_weights_from_train(np.array([0, 0, 0, 1]), np.array([0, 1]))
    {0: 0.6667, 1: 2.0}
    """
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): round(float(v), 4) for c, v in zip(classes, w)}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
