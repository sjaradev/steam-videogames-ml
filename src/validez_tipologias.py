"""Validez de las tipologías: referencia nula, coherencia de Jaccard y representaciones alternativas.

Responde al feedback de la fase 1:
  1. Compara los grupos contra una referencia nula (etiquetas al azar y datos con
     las columnas de tags permutadas, que conservan la frecuencia de cada tag pero
     rompen su co-ocurrencia).
  2. Mide la coherencia de cada tipología con el índice de Jaccard, la misma
     métrica con la que se planteó el problema.
  3. Prueba representaciones distintas a la binaria con distancia euclidiana:
     TF-IDF con distancia coseno y un modelo de pertenencia mixta (LDA).

Uso:
    python src/validez_tipologias.py --csv app_dataset_20260907_20260918.csv --out results/validez

Todo se entrena con el conjunto de entrenamiento y se mide en el de prueba.
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split

# Semilla para repetir las particiones y los muestreos; K fija las tipologías.
SEED = 42
K = 12
N_TAGS = 150
MIN_REVIEWS = 10
N_PAIRS = 4000        # pares muestreados por grupo para Jaccard
N_SIL = 5000          # juegos de prueba usados para la silueta
SEEDS = [42, 7, 21, 99, 123]


def parse_list(s):
    # El CSV guarda listas como texto. literal_eval las interpreta sin ejecutar código.
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if isinstance(v, dict):
        return list(v.keys())
    return list(v) if isinstance(v, (list, tuple)) else []


def load(csv):
    # Se consideran juegos lanzados, con suficientes reseñas y al menos un tag.
    df = pd.read_csv(csv, low_memory=False)
    df = df[(~df.coming_soon) & (df.total_reviews >= MIN_REVIEWS)].copy()
    df["tag_list"] = df.tags.map(parse_list)
    df["genre_list"] = df.genres.map(parse_list)
    df = df[df.tag_list.map(len) > 0]
    return df.reset_index(drop=True)


def binarize(tag_lists, vocab):
    # Cada fila es un juego y cada columna un tag: 1 si está presente, 0 si falta.
    # Los tags que no pertenecen al vocabulario se ignoran.
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(tag_lists), len(vocab)), dtype=np.float32)
    for r, tags in enumerate(tag_lists):
        for t in tags:
            j = idx.get(t)
            if j is not None:
                X[r, j] = 1.0
    return X


def pair_jaccard(B, i, j):
    # Para cada par de juegos: tags compartidos / tags presentes en alguno de los dos.
    inter = (B[i] * B[j]).sum(1)
    union = B[i].sum(1) + B[j].sum(1) - inter
    return inter / np.maximum(union, 1)


def random_jaccard(B, rng, n=20000):
    # Referencia de similitud sin agrupar; se descarta comparar un juego consigo mismo.
    i = rng.integers(0, len(B), n)
    j = rng.integers(0, len(B), n)
    keep = i != j
    return float(pair_jaccard(B, i[keep], j[keep]).mean())


def intra_jaccard(B, labels, rng):
    """Jaccard medio entre pares del mismo grupo. Devuelve (ponderado, por grupo)."""
    per, sizes = {}, {}
    for g in np.unique(labels):
        members = np.where(labels == g)[0]
        if len(members) < 2:
            continue
        # Muestrear pares evita calcular todas las combinaciones dentro del grupo.
        i = rng.choice(members, N_PAIRS)
        j = rng.choice(members, N_PAIRS)
        keep = i != j
        per[int(g)] = float(pair_jaccard(B, i[keep], j[keep]).mean())
        sizes[int(g)] = len(members)
    total = sum(sizes.values())
    # Los grupos con más juegos pesan más en el promedio final.
    weighted = sum(per[g] * sizes[g] for g in per) / total
    return weighted, per, sizes


def jaccard_distance_matrix(B):
    # La silueta necesita distancias: 0 indica tags iguales y 1, sin tags compartidos.
    # La matriz crece con el cuadrado de las filas; se usa solo una submuestra.
    inter = B @ B.T
    n = B.sum(1)
    union = n[:, None] + n[None, :] - inter
    D = 1.0 - inter / np.maximum(union, 1)
    np.fill_diagonal(D, 0.0)
    return D


def evaluate(name, labels_te, B_te, D_sil, sil_idx, rng, base):
    # Resume coherencia interna, separación y concentración para comparar agrupaciones.
    w, per, sizes = intra_jaccard(B_te, labels_te, rng)
    lab_s = labels_te[sil_idx]
    sil = silhouette_score(D_sil, lab_s, metric="precomputed") if len(set(lab_s)) > 1 else float("nan")
    share = max(sizes.values()) / sum(sizes.values())
    # lift_vs_azar indica cuántas veces supera el Jaccard medio a su referencia aleatoria.
    return {"modelo": name, "grupos": len(sizes), "jaccard_intra": w, "lift_vs_azar": w / base,
            "silueta_jaccard": float(sil), "grupo_mayor_pct": 100 * share,
            "jaccard_min": min(per.values()), "jaccard_max": max(per.values())}, per, sizes


def main():
    # --csv indica los datos de entrada y --out la carpeta de resultados.
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/validez")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load(a.csv)
    # Separa 70 % para entrenar, 15 % para validar y 15 % para evaluar en prueba.
    # Este script reserva validación, pero no la utiliza para ajustar los modelos.
    tr, tmp = train_test_split(df, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)
    # Elegir los tags solo con entrenamiento evita usar información de prueba.
    vocab = [t for t, _ in Counter(t for l in tr.tag_list for t in l).most_common(N_TAGS)]
    Xtr, Xte = binarize(tr.tag_list, vocab), binarize(te.tag_list, vocab)
    # Retira juegos sin tags reconocidos y conserva alineadas las filas de prueba.
    ktr, kte = Xtr.sum(1) > 0, Xte.sum(1) > 0
    Xtr, Xte, te = Xtr[ktr], Xte[kte], te[kte].reset_index(drop=True)
    info = {"juegos_filtrados": len(df), "train": int(len(Xtr)), "val": int(len(va)), "test": int(len(Xte)),
            "tags": N_TAGS, "k": K}

    base = random_jaccard(Xte, rng)
    info["jaccard_pares_al_azar"] = base
    # Reutiliza la misma submuestra y distancias para evaluar los modelos en datos reales.
    sil_idx = rng.choice(len(Xte), min(N_SIL, len(Xte)), replace=False)
    D_sil = jaccard_distance_matrix(Xte[sil_idx])

    rows, per_group = [], {}

    # --- Géneros de Steam (línea base del problema): pares que comparten género
    gcount = Counter(g for l in te.genre_list for g in l)
    gj = {}
    for g, _ in gcount.most_common(8):
        m = np.where(te.genre_list.map(lambda l: g in l).values)[0]
        i, j = rng.choice(m, N_PAIRS), rng.choice(m, N_PAIRS)
        keep = i != j
        gj[g] = float(pair_jaccard(Xte, i[keep], j[keep]).mean())
    info["jaccard_por_genero"] = gj
    # Para comparar particiones, asigna un único grupo según el primer género listado.
    # Esto es distinto del cálculo anterior, donde un juego puede compartir varios géneros.
    first = te.genre_list.map(lambda l: l[0] if l else "Sin género")
    top8 = [g for g, _ in gcount.most_common(8)]
    glab = pd.factorize(first.where(first.isin(top8), "Otros"))[0]
    r, _, _ = evaluate("Géneros de Steam (primer género)", glab, Xte, D_sil, sil_idx, rng, base)
    rows.append(r)

    # --- Referencia nula 1: etiquetas al azar con los mismos tamaños que K-Means
    km = KMeans(K, n_init=10, random_state=SEED).fit(Xtr)
    lab_km = km.predict(Xte)
    # Mezclar asignaciones conserva los tamaños, pero rompe su relación con los juegos.
    r, _, _ = evaluate("Nula: etiquetas al azar", rng.permutation(lab_km), Xte, D_sil, sil_idx, rng, base)
    rows.append(r)

    # --- Referencia nula 2: conserva frecuencias de tags y rompe coocurrencias originales.
    Ptr = np.column_stack([rng.permutation(Xtr[:, c]) for c in range(N_TAGS)])
    Pte = np.column_stack([rng.permutation(Xte[:, c]) for c in range(N_TAGS)])
    okp = Pte.sum(1) > 0
    Pte = Pte[okp]
    # Estos datos modificados necesitan su propia referencia y matriz de distancias.
    base_p = random_jaccard(Pte, rng)
    lab_p = KMeans(K, n_init=10, random_state=SEED).fit(Ptr).predict(Pte)
    pidx = rng.choice(len(Pte), min(N_SIL, len(Pte)), replace=False)
    r, _, _ = evaluate("Nula: K-Means sobre tags permutados", lab_p, Pte, jaccard_distance_matrix(Pte[pidx]),
                       pidx, rng, base_p)
    r["jaccard_azar_propio"] = base_p
    rows.append(r)

    # --- Modelo de la fase 1: binario + euclidiana
    r, per, sizes = evaluate("K-Means binario (euclidiana)", lab_km, Xte, D_sil, sil_idx, rng, base)
    rows.append(r)
    per_group["kmeans_binario"] = {"jaccard": per, "n": sizes}

    # --- TF-IDF normalizado: reduce el peso relativo de tags comunes.
    # K-Means sigue usando distancia euclidiana; no es K-Means esférico con coseno.
    # Se conserva el nombre histórico de la fila para compatibilidad con las figuras.
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=False).fit(Xtr)
    Ttr, Tte = tfidf.transform(Xtr).toarray(), tfidf.transform(Xte).toarray()
    kt = KMeans(K, n_init=10, random_state=SEED).fit(Ttr)
    lab_t = kt.predict(Tte)
    r, per, sizes = evaluate("K-Means TF-IDF (coseno)", lab_t, Xte, D_sil, sil_idx, rng, base)
    rows.append(r)
    per_group["kmeans_tfidf"] = {"jaccard": per, "n": sizes}
    # nombres: tags con mayor peso en cada centro
    names = {}
    for g in range(K):
        top = np.argsort(kt.cluster_centers_[g])[::-1][:4]
        names[g] = [vocab[t] for t in top]
    per_group["kmeans_tfidf"]["tags"] = names
    names = {}
    for g in range(K):
        top = np.argsort(km.cluster_centers_[g])[::-1][:4]
        names[g] = [vocab[t] for t in top]
    per_group["kmeans_binario"]["tags"] = names

    # --- LDA: pertenencia mixta
    lda = LatentDirichletAllocation(K, random_state=SEED, learning_method="batch", max_iter=30).fit(Xtr)
    Th = lda.transform(Xte)
    # Cada fila contiene pesos de tópicos; argmax permite comparar una partición única.
    lab_l = Th.argmax(1)
    r, per, sizes = evaluate("LDA (tópico dominante)", lab_l, Xte, D_sil, sil_idx, rng, base)
    rows.append(r)
    names = {g: [vocab[t] for t in np.argsort(lda.components_[g])[::-1][:4]] for g in range(K)}
    per_group["lda"] = {"jaccard": per, "n": sizes, "tags": names}
    # Los dos pesos mayores permiten resumir cuántos juegos mezclan tópicos.
    top2 = np.sort(Th, 1)[:, ::-1]
    info["lda"] = {"peso_medio_topico_dominante": float(top2[:, 0].mean()),
                   "pct_juegos_con_segundo_topico_mayor_0.2": float(100 * (top2[:, 1] > 0.2).mean()),
                   "pct_juegos_topico_dominante_menor_0.5": float(100 * (top2[:, 0] < 0.5).mean())}

    # --- Casos con pocos tags: un juego nuevo descrito con 2 o 3 etiquetas
    casos = [["Horror", "First-Person", "Atmospheric"], ["Visual Novel", "Story Rich"],
             ["Strategy", "Turn-Based Strategy", "Fantasy"], ["Shooter", "FPS", "Multiplayer"]]
    info["casos_pocos_tags"] = []
    for tags in casos:
        x = binarize([tags], vocab)
        # Aplica a cada ejemplo las mismas transformaciones aprendidas con train.
        gb = int(km.predict(x)[0])
        gt = int(kt.predict(tfidf.transform(x).toarray())[0])
        info["casos_pocos_tags"].append({"tags": tags, "binario": per_group["kmeans_binario"]["tags"][gb][:3],
                                         "tfidf": per_group["kmeans_tfidf"]["tags"][gt][:3]})

    # --- Estabilidad entre 5 semillas (ARI medio en prueba)
    def stability(Atr, Ate):
        # Compara las diez parejas de cinco ejecuciones; ARI no depende del número del grupo.
        labs = [KMeans(K, n_init=10, random_state=s).fit(Atr).predict(Ate) for s in SEEDS]
        aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
        return float(np.mean(aris)), float(np.min(aris))
    info["ari_5_semillas"] = {"kmeans_binario": stability(Xtr, Xte), "kmeans_tfidf": stability(Ttr, Tte)}
    info["ari_binario_vs_tfidf"] = float(adjusted_rand_score(lab_km, lab_t))
    info["silueta_euclidiana_binario"] = float(silhouette_score(Xte[sil_idx], lab_km[sil_idx]))
    info["silueta_coseno_tfidf"] = float(silhouette_score(Tte[sil_idx], lab_t[sil_idx], metric="cosine"))
    info["silueta_euclidiana_nula_permutada"] = float(silhouette_score(Pte[pidx], lab_p[pidx]))

    # Exporta una tabla por modelo y el detalle de tamaños, tags y coherencia por grupo.
    res = pd.DataFrame(rows)
    res.to_csv(out / "validez_modelos.csv", index=False)
    json.dump({"info": info, "por_grupo": per_group}, open(out / "validez_detalle.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    # Guarda la asignación de cada juego para poder cruzar los resultados entre modelos.
    te.assign(km_binario=lab_km, km_tfidf=lab_t, lda=lab_l)[["appid", "name", "km_binario", "km_tfidf", "lda"]] \
        .to_csv(out / "asignaciones_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(json.dumps(info, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
