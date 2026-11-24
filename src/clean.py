"""Limpeza e normalização do texto e dos rótulos."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------
# Rótulos
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = ["negativo", "neutro", "positivo"]


def rating_to_sentiment(rating: int) -> str:
    """Converte a nota de 1 a 5 em sentimento: 1–2 negativo, 3 neutro, 4–5 positivo.

    >>> [rating_to_sentiment(r) for r in range(1, 6)]
    ['negativo', 'negativo', 'neutro', 'positivo', 'positivo']
    """
    if rating <= 2:
        return "negativo"
    if rating == 3:
        return "neutro"
    return "positivo"


# ---------------------------------------------------------------------------
# Normalização (texto para o BERTimbau: mantém maiúsculas, acentos e pontuação)
# ---------------------------------------------------------------------------
# Sequências típicas de UTF-8 lido como Latin-1/CP1252 ("Ã©" no lugar de "é").
_MOJIBAKE = re.compile(r"Ã[\x80-\xbf¡-¿]|â€|Â[\x80-\xbf ]")
# Caracteres de controle e de largura zero (BOM, ZWSP etc.)
_INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ⁠﻿]")
_WHITESPACE = re.compile(r"\s+")


def has_mojibake(text: str) -> bool:
    return isinstance(text, str) and bool(_MOJIBAKE.search(text))


def fix_encoding(text: str) -> str:
    """Desfaz mojibake simples (UTF-8 decodificado como CP1252/Latin-1).

    >>> fix_encoding("NÃ£o gostei, pÃ©ssimo")
    'Não gostei, péssimo'
    >>> fix_encoding("NÃO GOSTEI")  # texto correto não é alterado
    'NÃO GOSTEI'
    """
    if not has_mojibake(text):
        return text
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not has_mojibake(fixed):
            return fixed
    return text


def normalize_text(text: str) -> str:
    """Corrige encoding, normaliza Unicode (NFC) e espaços/quebras de linha.

    NÃO converte para minúsculas nem remove acentos: o BERTimbau usado é *cased*.

    >>> normalize_text("  Produto\\r\\n\\n ótimo\\u200b!  ")
    'Produto ótimo!'
    """
    if not isinstance(text, str):
        return ""
    text = fix_encoding(text)
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = _WHITESPACE.sub(" ", text)  # inclui \r, \n e \t
    return text.strip()


def build_input(title: str, text: str) -> str:
    """Concatena título e texto em uma única entrada ("Título. Texto").

    >>> build_input("Ótimo", "Chegou rápido")
    'Ótimo. Chegou rápido'
    >>> build_input("Recomendo!", "Muito bom")
    'Recomendo! Muito bom'
    >>> build_input("", "Muito bom")
    'Muito bom'
    """
    title, text = (title or "").strip(), (text or "").strip()
    if not title:
        return text
    if not text:
        return title
    sep = " " if title[-1] in ".!?…" else ". "
    return f"{title}{sep}{text}"


def count_words(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0


# ---------------------------------------------------------------------------
# Texto para o baseline TF-IDF (minúsculas, sem pontuação, sem stopwords)
# ---------------------------------------------------------------------------
# Lista de stopwords do português (base: lista do NLTK), SEM as palavras de negação
# contraste e intensidade, que carregam sentimento ("não", "nem", "nunca", "mas", "muito"...).
_STOPWORDS_BASE = """
a à ao aos aquela aquelas aquele aqueles aquilo as às até com como da das de dela delas dele deles
depois do dos e é ela elas ele eles em entre era eram éramos essa essas esse esses esta está estamos
estão estar estas estava estavam estávamos este esteja estejam estejamos estes esteve estive estivemos
estiver estivera estiveram estivéramos estiverem estivermos estivesse estivessem estivéssemos estou eu
foi fomos for fora foram fôramos forem formos fosse fossem fôssemos fui há haja hajam hajamos hão havemos
haver hei houve houvemos houver houvera houverá houveram houvéramos houverão houverei houverem houveremos
houveria houveriam houveríamos houvermos houvesse houvessem houvéssemos isso isto já lhe lhes mais mas me
mesmo meu meus minha minhas muito na não nas nem no nos nós nossa nossas nosso nossos num numa o os ou
para pela pelas pelo pelos por qual quando que quem são se seja sejam sejamos sem ser será serão serei
seremos seria seriam seríamos seu seus só somos sou sua suas também te tem tém temos tenha tenham tenhamos
tenho terá terão terei teremos teria teriam teríamos teu teus teve tinha tinham tínhamos tive tivemos
tiver tivera tiveram tivéramos tiverem tivermos tivesse tivessem tivéssemos tu tua tuas um uma umas uns
você vocês vos pra pro q
""".split()
KEEP_WORDS = {"não", "nem", "nunca", "jamais", "nada", "nenhum", "nenhuma", "sem", "mas", "porém", "muito", "mais", "só"}
STOPWORDS_PT = frozenset(w for w in _STOPWORDS_BASE if w not in KEEP_WORDS)

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)  # sequências de letras (com acento)


def text_for_tfidf(text: str) -> str:
    """Versão do texto para o baseline: minúsculas, só palavras, sem stopwords.

    >>> text_for_tfidf("O produto NÃO chegou, e a loja é péssima!!! 10/10")
    'produto não chegou loja péssima'
    """
    tokens = _TOKEN.findall(text.lower()) if isinstance(text, str) else []
    return " ".join(t for t in tokens if t not in STOPWORDS_PT and len(t) > 1)


# ---------------------------------------------------------------------------
# Funil de limpeza
# ---------------------------------------------------------------------------
class CleaningFunnel:
    """Aplica etapas de limpeza registrando quantas linhas cada uma removeu."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rows = [{"etapa": "Base original", "removidas": 0, "restantes": len(df)}]

    def step(self, name: str, mask_keep: pd.Series) -> "CleaningFunnel":
        before = len(self.df)
        self.df = self.df[mask_keep]
        self.rows.append({"etapa": name, "removidas": before - len(self.df), "restantes": len(self.df)})
        return self

    def report(self) -> pd.DataFrame:
        rep = pd.DataFrame(self.rows)
        rep["% da base original"] = (100 * rep["removidas"] / self.rows[0]["restantes"]).round(2)
        return rep


def mask_conflicting_duplicates(df: pd.DataFrame, key: str, label: str) -> pd.Series:
    """Máscara (True = manter) para duplicados de conteúdo.

    - grupos com o mesmo `key` e o mesmo `label`: mantém só a primeira ocorrência;
    - grupos com o mesmo `key` e rótulos diferentes: remove todas as ocorrências
      (não há como saber qual rótulo é o correto).
    """
    n_labels = df.groupby(key)[label].transform("nunique")
    conflicting = n_labels > 1
    first = ~df.duplicated(subset=[key], keep="first")
    return first & ~conflicting


def contradictory_mask(df: pd.DataFrame) -> pd.Series:
    """True para avaliações com nota extrema contradita pela recomendação.

    Nota 1 com "recomendaria" (Yes) ou nota 5 com "não recomendaria" (No).
    """
    rating, rec = df["overall_rating"], df["recommend_to_a_friend"]
    return ((rating == 1) & (rec == "Yes")) | ((rating == 5) & (rec == "No"))
