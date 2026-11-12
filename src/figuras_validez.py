"""Figuras de la validez de las tipologías (Jaccard por modelo y por tipología).

Uso:
    python src/figuras_validez.py --res results/validez --out results/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AZUL, GRIS = "#245b96", "#9aa5b1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/validez")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(res / "validez_modelos.csv").set_index("modelo")
    d = json.load(open(res / "validez_detalle.json", encoding="utf-8"))
    base = d["info"]["jaccard_pares_al_azar"]
    nula = r.loc["Nula: K-Means sobre tags permutados", "jaccard_intra"]

    # 13: Jaccard medio por modelo
    filas = [("Pares al azar", base, GRIS),
             ("Géneros de Steam", r.loc["Géneros de Steam (primer género)", "jaccard_intra"], GRIS),
             ("K-Means sobre\ntags permutados (nula)", nula, GRIS),
             ("LDA\n(tópico dominante)", r.loc["LDA (tópico dominante)", "jaccard_intra"], AZUL),
             ("K-Means TF-IDF\n(coseno)", r.loc["K-Means TF-IDF (coseno)", "jaccard_intra"], AZUL),
             ("K-Means binario\n(euclidiana)", r.loc["K-Means binario (euclidiana)", "jaccard_intra"], AZUL)][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.barh([f[0] for f in filas], [f[1] for f in filas], color=[f[2] for f in filas])
    for y, f in enumerate(filas):
        ax.text(f[1] + 0.003, y, f"{f[1]:.3f}".replace(".", ","), va="center")
    ax.axvline(base, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 0.25)
    ax.set_xlabel("Jaccard medio entre pares del mismo grupo (conjunto de prueba)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "13_jaccard_modelos.png", dpi=200)

    # 14: Jaccard de cada tipología del K-Means binario
    g = d["por_grupo"]["kmeans_binario"]
    ids = sorted(g["jaccard"], key=lambda k: g["jaccard"][k])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh([" / ".join(g["tags"][k][:3]) for k in ids], [g["jaccard"][k] for k in ids], color=AZUL)
    ax.axvline(base, color="k", ls="--", lw=0.8, label="pares al azar")
    ax.axvline(nula, color="#b04a2f", ls=":", lw=1, label="referencia nula")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("Jaccard medio dentro de la tipología (conjunto de prueba)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "14_jaccard_tipologias.png", dpi=200)


if __name__ == "__main__":
    main()
