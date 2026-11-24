"""Download reproduzível do corpus B2W-Reviews01.

Fonte oficial: https://github.com/americanas-tech/b2w-reviews01 (antigo b2wdigital/b2w-reviews01)
Licença: CC BY-NC-SA 4.0 (crédito à B2W Digital, uso não comercial, compartilhamento pela mesma licença).

O arquivo é baixado de um commit fixo do repositório e conferido por SHA-256, de modo
que qualquer pessoa que rode o script obtém exatamente os mesmos bytes.

Uso:
    python -m src.download            # baixa para data/raw/ (se ainda não existir)
    python -m src.download --force    # baixa de novo
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

COMMIT = "4639429ec698d7821fc99a0bc665fa213d9fcd5a"
FILENAME = "B2W-Reviews01.csv"
URL = f"https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/{COMMIT}/{FILENAME}"
SHA256 = "821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calcula o SHA-256 de um arquivo lendo em blocos."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _show(path: Path) -> str:
    """Caminho relativo à raiz do projeto (evita expor caminhos locais nas saídas do notebook)."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def download(dest_dir: Path = RAW_DIR, force: bool = False, timeout: int = 120) -> Path:
    """Baixa o CSV para `dest_dir` e valida o hash. Retorna o caminho do arquivo."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / FILENAME
    if dest.exists() and not force:
        if sha256_of(dest) == SHA256:
            print(f"Arquivo já existe e confere com o SHA-256: {_show(dest)}")
            return dest
        print("Arquivo existente com hash diferente; baixando novamente.")

    tmp = dest.with_suffix(".part")
    print(f"Baixando {URL}")
    with requests.get(URL, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    digest = sha256_of(tmp)
    if digest != SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 inesperado: {digest} (esperado {SHA256})")
    tmp.replace(dest)
    print(f"OK: {_show(dest)} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="baixa mesmo se o arquivo já existir")
    args = parser.parse_args()
    download(force=args.force)
