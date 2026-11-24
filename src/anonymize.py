"""Anonimização: remoção de colunas do avaliador e mascaramento de dados pessoais no texto.

Os padrões são propositalmente conservadores (exigem a formatação típica de cada dado)
para não mascarar preços, modelos de produto ou notas por engano.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# Colunas que identificam (ou ajudam a identificar) quem escreveu a avaliação.
# Não são usadas pela tarefa, então são descartadas logo na carga.
REVIEWER_COLUMNS = ["reviewer_id", "reviewer_birth_year", "reviewer_gender", "reviewer_state"]

# A ordem importa: padrões mais específicos primeiro (CNPJ antes de CPF, e-mail antes de URL).
PII_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "URL": re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE),
    "CNPJ": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/\d{4}-?\d{2}\b"),
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "CEP": re.compile(r"\b\d{5}-\d{3}\b"),
    # (11) 91234-5678 | 11 91234-5678 | 91234-5678 | 3333-4444 | +55 11 3333 4444
    "TELEFONE": re.compile(r"(?:\+55\s?)?(?:\(\d{2}\)\s?|\b\d{2}\s)?\b9?\d{4}[-\s]\d{4}\b"),
    # Código de rastreio dos Correios, ex.: PP075161001BR (identifica a encomenda/destinatário)
    "RASTREIO": re.compile(r"\b[A-Z]{2}\d{9}[A-Z]{2}\b"),
}


def drop_reviewer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove as colunas de identificação do avaliador (as que existirem)."""
    return df.drop(columns=[c for c in REVIEWER_COLUMNS if c in df.columns])


def mask_pii(text: str, counter: Counter | None = None) -> str:
    """Substitui dados pessoais por marcadores como [EMAIL] e [CPF].

    >>> mask_pii("ligue (11) 91234-5678 ou mande para joao@ex.com.br")
    'ligue [TELEFONE] ou mande para [EMAIL]'
    >>> mask_pii("CPF 123.456.789-00, CEP 01310-100")
    'CPF [CPF], CEP [CEP]'
    >>> mask_pii("Custou R$ 1.299,90 e chegou em 5 dias")  # preços não são afetados
    'Custou R$ 1.299,90 e chegou em 5 dias'
    """
    if not isinstance(text, str) or not text:
        return text
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label}]", text)
        if counter is not None and n:
            counter[label] += n
    return text


def mask_pii_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica `mask_pii` nas colunas indicadas.

    Retorna o DataFrame mascarado e uma tabela com o número de ocorrências por tipo e coluna.
    """
    df = df.copy()
    rows = []
    for col in columns:
        counter: Counter = Counter()
        df[col] = df[col].map(lambda s: mask_pii(s, counter))
        rows.append({"coluna": col, **{k: counter.get(k, 0) for k in PII_PATTERNS}})
    report = pd.DataFrame(rows).set_index("coluna")
    report.loc["total"] = report.sum()
    return df, report
