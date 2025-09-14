
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import joblib
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

def norm_path(p: str) -> str:
    return os.path.normpath(p)

# ---------- Robust key normalization & aliasing ----------
def _knorm(s):
    if not isinstance(s, str):
        s = str(s)
    # remove underscores, spaces (incl. non‑breaking), hyphens; lowercase
    return (
        s.replace('\u00A0',' ')
         .strip()
         .lower()
         .replace('_','')
         .replace('-','')
         .replace(' ','')
    )

ALIASES = {
    'id': ['id','msgid','messageid'],
    'parent_id': ['parent_id','parentid','replyto','inreplyto'],
    'target_emb': ['target_emb','targetembedding','targetemb','embalvo','embtarget'],
    'emb': ['emb','embedding','currentemb','embatual','x','features'],
    'parent_emb': ['parent_emb','parentembedding','parentemb','embparent'],
    'label': ['label','y','labelenc','label_encoded'],
    'parent_label': ['parent_label','parentlabel','parentlabelenc','parent_label_enc'],
    'thread': ['thread','threadid','conversationid','targetid','target_id'],
    'grau_distancia': ['grau_distancia','graudistancia','depth','grau','distance','level'],
}

REQUIRED_STD_KEYS = ['id','parent_id','target_emb','emb','parent_emb','label','parent_label','thread','grau_distancia']

def standardize_keys(d):
    """Return a new dict with standardized keys using ALIASES and robust normalization.
       Raises KeyError with helpful message if something cannot be mapped.
    """
    orig_keys = list(d.keys())
    norm_to_orig = {}
    for k in orig_keys:
        norm_to_orig[_knorm(k)] = k

    std = {}
    missing_std = []
    used = {}

    for std_key in REQUIRED_STD_KEYS:
        candidates = ALIASES[std_key]
        found = None
        for cand in candidates:
            kn = _knorm(cand)
            if kn in norm_to_orig:
                found = norm_to_orig[kn]
                break
        if found is None:
            # Try fuzzy: look for close names present
            # gather any existing keys that share big substring
            hints = [ok for ok in orig_keys if _knorm(ok).find(_knorm(std_key)) >= 0]
            missing_std.append((std_key, hints))
        else:
            std[std_key] = d[found]
            used[std_key] = found

    if missing_std:
        lines = ["\nChaves ausentes após normalização/aliasing:"]
        for std_key, hints in missing_std:
            if hints:
                lines.append(f" - '{std_key}' (possíveis no arquivo: {hints})")
            else:
                lines.append(f" - '{std_key}' (nenhuma pista; chaves disponíveis: {list(d.keys())})")
        raise KeyError("\n".join(lines))

    # small sanity: lengths align
    n = None
    for k in ['emb','target_emb','label','parent_label','thread','grau_distancia']:
        arr = np.asarray(std[k])
        n = len(arr) if n is None else n
        if len(arr) != n:
            raise ValueError(f"Tamanho inconsistente: '{k}' tem len={len(arr)} mas esperado={n}")
    # parent_emb may have same n, or be optional shape; we won't enforce strict shape here

    # Debug log (helps user see which original keys were mapped)
    std['_mapped_keys_debug'] = used  # keep for later prints
    return std

def ensure_2d_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def concat_features(emb, target_emb, parent_label_col, parent_emb=None, usar_parent_emb=False):
    parts = [emb, target_emb, parent_label_col]
    if usar_parent_emb and parent_emb is not None:
        parts.append(parent_emb)
    return np.concatenate(parts, axis=1)

def sanitize_ids(arr):
    arr = np.asarray(arr, dtype=object)
    out = []
    for v in arr:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out.append(None)
        else:
            out.append(v)
    return np.asarray(out, dtype=object)

def sanitize_grau(arr):
    a = np.asarray(arr, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(a).astype(int)

def treinar_modelo_base(dados_treino, n_estimators=200, random_state=42, usar_parent_emb=False):
    emb = np.asarray(dados_treino['emb'])
    target_emb = np.asarray(dados_treino['target_emb'])
    parent_label = ensure_2d_col(dados_treino['parent_label'])
    parent_emb = np.asarray(dados_treino['parent_emb']) if 'parent_emb' in dados_treino else None
    y = np.asarray(dados_treino['label']).ravel()

    X = concat_features(emb, target_emb, parent_label, parent_emb, usar_parent_emb=usar_parent_emb)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X, y)
    return clf

def inferir_sequencial_ids(clf_seq, dados_treino, dados_teste, usar_parent_emb=False):
    emb_tr = np.asarray(dados_treino['emb'])
    tar_tr = np.asarray(dados_treino['target_emb'])
    par_tr = ensure_2d_col(dados_treino['parent_label'])
    pem_tr = np.asarray(dados_treino['parent_emb']) if 'parent_emb' in dados_treino else None
    y_tr   = np.asarray(dados_treino['label']).ravel()
    thr_tr = sanitize_ids(dados_treino['thread'])
    grau_tr= sanitize_grau(dados_treino['grau_distancia'])

    emb_te = np.asarray(dados_teste['emb'])
    tar_te = np.asarray(dados_teste['target_emb'])
    par_te = ensure_2d_col(dados_teste['parent_label'])
    pem_te = np.asarray(dados_teste['parent_emb']) if 'parent_emb' in dados_teste else None
    y_te   = np.asarray(dados_teste['label']).ravel()
    thr_te = sanitize_ids(dados_teste['thread'])
    grau_te= sanitize_grau(dados_teste['grau_distancia'])

    ids_tr = sanitize_ids(dados_treino['id'])
    ids_te = sanitize_ids(dados_teste['id'])
    pid_tr = sanitize_ids(dados_treino['parent_id'])
    pid_te = sanitize_ids(dados_teste['parent_id'])

    idx_tr_by_id = {ids_tr[i]: i for i in range(len(ids_tr)) if ids_tr[i] is not None}
    idx_te_by_id = {ids_te[i]: i for i in range(len(ids_te)) if ids_te[i] is not None}

    order_tr = np.argsort(grau_tr, kind='stable')
    y_pred_seq_train = np.full(shape=(len(emb_tr),), fill_value=-1, dtype=int)

    for i in order_tr:
        parent_feat_val = par_tr[i, 0]
        p_id = pid_tr[i]
        if (p_id is not None) and (p_id in idx_tr_by_id):
            j_parent = idx_tr_by_id[p_id]
            if y_pred_seq_train[j_parent] != -1:
                parent_feat_val = y_pred_seq_train[j_parent]

        X_i = concat_features(
            emb_tr[i:i+1],
            tar_tr[i:i+1],
            np.array([[parent_feat_val]], dtype=float),
            pem_tr[i:i+1] if usar_parent_emb and (pem_tr is not None) else None,
            usar_parent_emb=usar_parent_emb
        )
        y_pred_i = clf_seq.predict(X_i)[0]
        y_pred_seq_train[i] = y_pred_i

    order_te = np.argsort(grau_te, kind='stable')
    y_pred_seq_test = np.full(shape=(len(emb_te),), fill_value=-1, dtype=int)
    info_parent_feature_test = []

    for i in order_te:
        parent_feat_val = par_te[i, 0]
        parent_source = "real_parent_label"
        p_id = pid_te[i]
        if (p_id is not None):
            j_parent_te = idx_te_by_id.get(p_id, None)
            if j_parent_te is not None and y_pred_seq_test[j_parent_te] != -1:
                parent_feat_val = y_pred_seq_test[j_parent_te]
                parent_source = "predicted_from_test_parent"
            else:
                j_parent_tr = idx_tr_by_id.get(p_id, None)
                if j_parent_tr is not None and y_pred_seq_train[j_parent_tr] != -1:
                    parent_feat_val = y_pred_seq_train[j_parent_tr]
                    parent_source = "predicted_from_train_parent"

        X_i = concat_features(
            emb_te[i:i+1],
            tar_te[i:i+1],
            np.array([[parent_feat_val]], dtype=float),
            pem_te[i:i+1] if usar_parent_emb and (pem_te is not None) else None,
            usar_parent_emb=usar_parent_emb
        )
        y_pred_i = clf_seq.predict(X_i)[0]
        y_pred_seq_test[i] = y_pred_i

        info_parent_feature_test.append({
            "index_test": int(i),
            "id": ids_te[i] if ids_te[i] is not None else "None",
            "parent_id": p_id if p_id is not None else "None",
            "thread": str(thr_te[i]),
            "grau": int(grau_te[i]),
            "parent_feature_used": float(parent_feat_val),
            "parent_feature_source": parent_source
        })

    return y_pred_seq_test, y_pred_seq_train, info_parent_feature_test

def relatorio_geral(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    rep = classification_report(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "classification_report": rep}

def relatorio_por_grau(y_true, y_pred, graus):
    graus = np.asarray(graus)
    out = {}
    for g in np.unique(graus):
        mask = (graus == g)
        if mask.sum() == 0:
            continue
        yg = y_true[mask]
        pg = y_pred[mask]
        out[int(g)] = relatorio_geral(yg, pg)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", type=str, default="input/input_conjuntoDeDados_treinamento.joblib")
    ap.add_argument("--teste",  type=str, default="input/input_conjuntoDeDados_teste.joblib")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--usar_parent_emb", type=int, default=0)
    args = ap.parse_args()

    print(">> Carregando dados...")
    dados_treino_raw = joblib.load(norm_path(args.treino))
    dados_teste_raw  = joblib.load(norm_path(args.teste))

    # Standardize/alias keys here (robust against stray spaces/variants)
    dados_treino = standardize_keys(dados_treino_raw)
    dados_teste  = standardize_keys(dados_teste_raw)

    # Print mapped keys for transparency
    print(">> Mapeamento de chaves (treino):", dados_treino.get('_mapped_keys_debug', {}))
    print(">> Mapeamento de chaves (teste):",  dados_teste.get('_mapped_keys_debug', {}))

    usar_parent_emb = bool(args.usar_parent_emb)

    print(">> Treinando MODELO BASE...")
    clf_base = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
    X_tr_base = concat_features(
        np.asarray(dados_treino['emb']),
        np.asarray(dados_treino['target_emb']),
        ensure_2d_col(dados_treino['parent_label']),
        np.asarray(dados_treino['parent_emb']) if usar_parent_emb else None,
        usar_parent_emb=usar_parent_emb
    )
    y_tr = np.asarray(dados_treino['label']).ravel()
    clf_base.fit(X_tr_base, y_tr)

    X_te_base = concat_features(
        np.asarray(dados_teste['emb']),
        np.asarray(dados_teste['target_emb']),
        ensure_2d_col(dados_teste['parent_label']),
        np.asarray(dados_teste['parent_emb']) if usar_parent_emb else None,
        usar_parent_emb=usar_parent_emb
    )
    y_test_base = clf_base.predict(X_te_base)

    print(">> Treinando MODELO SEQUENCIAL...")
    clf_seq = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
    clf_seq.fit(X_tr_base, y_tr)  # treino igual ao base

    print(">> Inferindo SEQUENCIALMENTE no TESTE...")
    y_pred_seq_test, y_pred_seq_train, parent_diag = inferir_sequencial_ids(
        clf_seq, dados_treino, dados_teste, usar_parent_emb=usar_parent_emb
    )

    y_true_test = np.asarray(dados_teste['label']).ravel()
    graus_test  = sanitize_grau(dados_teste['grau_distancia'])

    rel_base_geral = relatorio_geral(y_true_test, y_test_base)
    rel_seq_geral  = relatorio_geral(y_true_test, y_pred_seq_test)

    rel_base_grau = relatorio_por_grau(y_true_test, y_test_base, graus_test)
    rel_seq_grau  = relatorio_por_grau(y_true_test, y_pred_seq_test, graus_test)

    # --- Counts por grau no conjunto de TESTE ---
    _unique_graus, _counts = np.unique(graus_test, return_counts=True)
    test_counts_by_grau = {int(g): int(c) for g, c in zip(_unique_graus, _counts)}

    print("\n==================== COMPARATIVO GERAL ====================")
    print(f"BASE       | Accuracy: {rel_base_geral['accuracy']:.3f} | F1-macro: {rel_base_geral['f1_macro']:.3f}")
    print(f"SEQUENCIAL | Accuracy: {rel_seq_geral['accuracy']:.3f} | F1-macro: {rel_seq_geral['f1_macro']:.3f}")

    print("\n----- Classification Report (BASE) -----")
    print(rel_base_geral['classification_report'])

    print("\n----- Classification Report (SEQUENCIAL) -----")
    print(rel_seq_geral['classification_report'])

    print("\n==================== DISTRIBUIÇÃO DE CASOS NO TESTE POR GRAU ====================")
    for g in sorted(test_counts_by_grau.keys()):
        print(f"Grau {g}: {test_counts_by_grau[g]} casos")
    print("\n==================== POR GRAU (TESTE) ====================")
    todos_graus = sorted(set(list(rel_base_grau.keys()) + list(rel_seq_grau.keys())))
    for g in todos_graus:
        rb = rel_base_grau.get(g, None)
        rs = rel_seq_grau.get(g, None)
        if rb:
            print(f"\n[Grau {g}] BASE       | Acc: {rb['accuracy']:.3f} | F1-macro: {rb['f1_macro']:.3f}")
        if rs:
            print(f"[Grau {g}] SEQUENCIAL | Acc: {rs['accuracy']:.3f} | F1-macro: {rs['f1_macro']:.3f}")

    origem_counts = defaultdict(int)
    for d in parent_diag:
        origem_counts[d["parent_feature_source"]] += 1
    total_test = len(parent_diag)
    print("\n-------------------- Diagnóstico da feature 'parent_label' no TESTE --------------------")
    for k, v in origem_counts.items():
        print(f"{k:>28s}: {v:4d} ({100.0*v/total_test:5.1f}%)")

    os.makedirs("output", exist_ok=True)
    import json
    resultados = {
        "test_counts_by_grau": test_counts_by_grau,
        "y_test_true": y_true_test,
        "y_test_base": y_test_base,
        "y_test_sequencial": y_pred_seq_test,
        "rel_base_geral": {k: v for k, v in rel_base_geral.items() if k != "classification_report"},
        "rel_seq_geral":  {k: v for k, v in rel_seq_geral.items()  if k != "classification_report"},
        "rel_base_grau":  rel_base_grau,
        "rel_seq_grau":   rel_seq_grau,
        "diagnostico_parent_feature": parent_diag,
    }
    joblib.dump(resultados, "output/resultados_modelagem_sequencial.joblib")
    with open("output/relatorio_geral.txt", "w", encoding="utf-8") as f:
        f.write("=== COMPARATIVO GERAL ===\n")
        f.write(f"BASE       | Accuracy: {rel_base_geral['accuracy']:.4f} | F1-macro: {rel_base_geral['f1_macro']:.4f}\n")
        f.write(f"SEQUENCIAL | Accuracy: {rel_seq_geral['accuracy']:.4f} | F1-macro: {rel_seq_geral['f1_macro']:.4f}\n\n")
        f.write("----- Classification Report (BASE) -----\n")
        f.write(rel_base_geral['classification_report'])
        f.write("\n\n----- Classification Report (SEQUENCIAL) -----\n")
        f.write(rel_seq_geral['classification_report'])
        f.write("\n\n=== DISTRIBUIÇÃO DE CASOS NO TESTE POR GRAU ===\n")
        for g in sorted(test_counts_by_grau.keys()):
            f.write(f"Grau {g}: {test_counts_by_grau[g]} casos\n")
        f.write("\n=== POR GRAU ===\n")
        for g in todos_graus:
            rb = rel_base_grau.get(g, None)
            rs = rel_seq_grau.get(g, None)
            if rb:
                f.write(f"\n[Grau {g}] BASE       | Acc: {rb['accuracy']:.4f} | F1-macro: {rb['f1_macro']:.4f}\n")
            if rs:
                f.write(f"[Grau {g}] SEQUENCIAL | Acc: {rs['accuracy']:.4f} | F1-macro: {rs['f1_macro']:.4f}\n")

        f.write("\n\n--- Diagnóstico da origem da feature 'parent_label' no TESTE ---\n")
        for k, v in origem_counts.items():
            f.write(f"{k:>28s}: {v:4d} ({100.0*v/total_test:5.1f}%)\n")

    print("\nArquivos gerados em ./output :")
    print(" - resultados_modelagem_sequencial.joblib")
    print(" - relatorio_geral.txt")

if __name__ == "__main__":
    main()
