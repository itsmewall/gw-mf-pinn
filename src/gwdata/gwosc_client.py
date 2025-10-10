"""
gwosc_client.py — Compatível com GWOSC API v2.

Funções:
 - list_runs(): lista runs públicas
 - download_event(event): baixa strain files H1/L1/V1 do evento
 - verify_integrity(path): valida HDF5 e imprime SHA
 - load_strain(path): retorna (t, strain)

Requisitos: requests, h5py, numpy
"""

import os
import hashlib
import requests
import h5py
import numpy as np
from typing import List, Tuple
from urllib.parse import urlparse

GWOSC = "https://gwosc.org"

# -------------------------------
# Runs
# -------------------------------
def list_runs() -> List[str]:
    """Lista runs públicas (S5, S6, O1, O2, O3...)."""
    url = f"{GWOSC}/api/v2/runs"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        runs = [run.get("name", "") for run in data.get("results", []) if run.get("name")]
        print(f"[GWOSC] {len(runs)} runs encontrados (ex: {runs[:3]})")
        return runs
    except Exception as e:
        print(f"[ERRO] Falha ao listar runs: {e}")
        return []

# -------------------------------
# Download por evento
# -------------------------------
def download_event(event: str = "GW150914", out_dir: str = "data/raw") -> bool:
    """
    Baixa os strain-files de um evento: /api/v2/events/<event>/strain-files
    Cada item costuma trazer hdf5_url, gwf_url e detail_url (API v2).
    """
    os.makedirs(out_dir, exist_ok=True)
    api = f"{GWOSC}/api/v2/events/{event}/strain-files"
    saved = []

    try:
        r = requests.get(api, timeout=20)
        r.raise_for_status()
        payload = r.json()

        results = payload.get("results") or []
        if not results:
            print(f"[GWOSC] Nenhum strain file encontrado para {event}.")
            return False

        print(f"[GWOSC] {len(results)} arquivos disponíveis para {event}.")

        for item in results:
            # Prioriza HDF5 (mais simples de ler com h5py)
            url = item.get("hdf5_url") or item.get("gwf_url")
            if not url:
                # fallback: às vezes só detail_url; poderíamos buscar o detail pra obter hdf5_url
                detail = item.get("detail_url")
                if detail:
                    try:
                        rd = requests.get(detail, timeout=20)
                        rd.raise_for_status()
                        djson = rd.json()
                        url = djson.get("hdf5_url") or djson.get("gwf_url")
                    except Exception as e:
                        print(f"[WARN] detail_url falhou ({e}); pulando item.")
                if not url:
                    print("[ERRO] Nenhum link de download no item; pulando.")
                    continue

            # Nome do arquivo: derive da URL (evita 'L1.hdf5' genérico)
            fname = os.path.basename(urlparse(url).path) or (item.get("detector", "det") + ".hdf5")
            fpath = os.path.join(out_dir, fname)

            print(f"[GWOSC] Baixando {fname} ...")
            _download_file(url, fpath)
            saved.append(fpath)

        if saved:
            print("[GWOSC] Arquivos salvos:")
            for p in saved:
                print(f"  - {p}")
            return True

        print(f"[GWOSC] Nada foi salvo para {event}.")
        return False

    except Exception as e:
        print(f"[ERRO] Falha ao baixar {event}: {e}")
        return False

def _download_file(url: str, dest: str, chunk_size: int = 1 << 14):
    """Baixa binário com progresso simples."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if total:
                    done += len(chunk)
                    print(f"\r  [Baixando] {done/total:.1%}", end="")
        print("\r  [OK] salvo.")

# -------------------------------
# Verificação e leitura
# -------------------------------
def verify_integrity(file_path: str) -> bool:
    """Abre HDF5 e imprime chaves + SHA curta."""
    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
        with open(file_path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()[:12]
        print(f"[OK] {file_path} | SHA256={sha} | Keys={keys}")
        return True
    except Exception as e:
        print(f"[ERRO] HDF5 inválido: {e}")
        return False

def load_strain(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (t, strain) a partir de /strain/Strain."""
    try:
        with h5py.File(file_path, "r") as f:
            dset = f["strain"]["Strain"]
            strain = dset[:]
            dt = dset.attrs.get("Xspacing")
        t = np.arange(0, len(strain) * dt, dt)
        return t, strain
    except Exception as e:
        print(f"[ERRO] Falha ao ler strain: {e}")
        return None, None

# -------------------------------
# Main de teste
# -------------------------------
if __name__ == "__main__":
    print("[GWOSC] Iniciando teste básico de conexão...\n")
    _ = list_runs()
    print()

    event = "GW150914"
    print(f"[GWOSC] Testando download do evento {event}...\n")
    ok = download_event(event)
    if ok:
        print(f"\n[GWOSC] Download concluído com sucesso para {event}.")
    else:
        print(f"\n[GWOSC] Falha ao baixar o evento {event}.")
