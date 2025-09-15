
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Um ÚNICO modelo; DUAS testagens (Base vs. Sequencial)

Visão geral
-----------
- Treino: treina-se apenas UM classificador, usando parent_label REAL no treino (como no Base).
- Teste Base: usa o mesmo classificador, com parent_label REAL do TESTE nas features.
- Teste Sequencial: usa o MESMO classificador, porém encadeando previsões do pai no TESTE (sem rótulo real),
  grau a grau.

Regras no TESTE (Sequencial)
----------------------------
- Grau 0: NÃO prever (sem parent); loga "skipped_degree0".
- Grau 1: prever assumindo parent_feature = +1 ("Concorda"); loga "assumed_concorda_degree1".
- Grau >= 2: prever somente se o pai (no TESTE) JÁ FOI previsto; senão pular e logar
  "skipped_missing_parent_prediction".

Observações
-----------
- O sequencial NUNCA usa parent_label REAL no teste e NUNCA usa pai do TREINO.
- As métricas do Sequencial são computadas APENAS sobre as amostras efetivamente previstas.
- O relatório inclui contagem por grau, cobertura por grau, diagnóstico da origem da feature do pai, e
  threads com graus fora do esperado (ver Exceção III).
"""

import joblib
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# =====================
# Helpers
# =====================

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

def sanitize_grau(arr):
    a = np.asarray(arr, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(a).astype(int)

# =====================
# Treino Base
# =====================

def treinar_modelo_base(dados_treino, n_estimators=200, random_state=42, usar_parent_emb=False):
    """Treina UM único RandomForest com as features do treino.

    - Features: embedding do comentário atual + embedding do alvo + parent_label REAL do treino
    - Target: label do comentário atual
    """
    # Extrai as matrizes/colunas necessárias do dicionário de treino
    emb = dados_treino['emb']
    tar = dados_treino['target_emb']
    par = ensure_2d_col(dados_treino['parent_label'])  # parent_label REAL no treino
    y   = dados_treino['label']

    # parent_emb é opcional (controlado pelo parâmetro usar_parent_emb)
    pem = dados_treino.get('parent_emb', None) if usar_parent_emb else None

    # Concatena as features selecionadas em um único array 2D
    X = concat_features(emb, tar, par, pem, usar_parent_emb=usar_parent_emb)

    # Define e treina o classificador
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X, y)
    return clf

# =====================
# Inferência Sequencial (TEST-ONLY chaining)
# =====================

def inferir_sequencial_ids(clf, dados_teste, usar_parent_emb=False):
    """Encadeia previsões no TESTE, grau a grau, sem usar rótulos reais.

    Regras de encadeamento no teste:
    - Grau 0: não prever (sem pai definido)
    - Grau 1: prever assumindo parent_label = +1 ("Concorda")
    - Grau >= 2: prever somente se o pai (no TESTE) já foi previsto
    """
    # --- arrays do TESTE ---
    emb_te = dados_teste['emb']
    tar_te = dados_teste['target_emb']
    pem_te = dados_teste.get('parent_emb', None) if usar_parent_emb else None
    ids_te = dados_teste['id']
    pid_te = dados_teste['parent_id']
    thr_te = dados_teste['thread']
    grau_te = sanitize_grau(dados_teste['grau_distancia'])

    # dicionário id->índice no TESTE
    idx_te_by_id = {ids_te[i]: i for i in range(len(ids_te)) if ids_te[i] is not None}

    # buffers
    y_pred_seq_test = np.full(shape=(len(emb_te),), fill_value=-1, dtype=int)
    info_parent_feature_test = []

    # processar em ordem de grau
    for g in sorted(np.unique(grau_te)):
        idxs = np.where(grau_te == g)[0]

        if g == 0:
            # NÃO prever grau 0
            for i in idxs:
                info_parent_feature_test.append({
                    "index_test": int(i),
                    "id": ids_te[i] if ids_te[i] is not None else "None",
                    "parent_id": pid_te[i] if pid_te[i] is not None else "None",
                    "thread": str(thr_te[i]),
                    "grau": int(grau_te[i]),
                    "parent_feature_used": None,
                    "parent_feature_source": "skipped_degree0"
                })
            continue

        if g == 1:
            # Prever grau 1 assumindo parent_feature = +1 ("Concorda")
            for i in idxs:
                parent_feat_val = 1.0
                parent_source = "assumed_concorda_degree1"
                X_i = concat_features(
                    emb_te[i:i+1],
                    tar_te[i:i+1],
                    np.array([[parent_feat_val]], dtype=float),
                    pem_te[i:i+1] if (usar_parent_emb and pem_te is not None) else None,
                    usar_parent_emb=usar_parent_emb
                )
                y_pred_i = clf.predict(X_i)[0]
                y_pred_seq_test[i] = y_pred_i
                info_parent_feature_test.append({
                    "index_test": int(i),
                    "id": ids_te[i] if ids_te[i] is not None else "None",
                    "parent_id": pid_te[i] if pid_te[i] is not None else "None",
                    "thread": str(thr_te[i]),
                    "grau": int(grau_te[i]),
                    "parent_feature_used": float(parent_feat_val),
                    "parent_feature_source": parent_source
                })
            continue

        # g >= 2: prever somente se o pai no TESTE já foi previsto
        for i in idxs:
            p_id = pid_te[i]
            parent_feat_val = None
            parent_source = "skipped_missing_parent_prediction"

            if p_id is not None:
                j_parent = idx_te_by_id.get(p_id, None)
                if (j_parent is not None) and (y_pred_seq_test[j_parent] != -1):
                    parent_feat_val = y_pred_seq_test[j_parent]
                    parent_source = "predicted_from_test_parent"

            if parent_feat_val is None:
                # pular e logar
                info_parent_feature_test.append({
                    "index_test": int(i),
                    "id": ids_te[i] if ids_te[i] is not None else "None",
                    "parent_id": p_id if p_id is not None else "None",
                    "thread": str(thr_te[i]),
                    "grau": int(grau_te[i]),
                    "parent_feature_used": None,
                    "parent_feature_source": parent_source
                })
                continue

            X_i = concat_features(
                emb_te[i:i+1],
                tar_te[i:i+1],
                np.array([[parent_feat_val]], dtype=float),
                pem_te[i:i+1] if (usar_parent_emb and pem_te is not None) else None,
                usar_parent_emb=usar_parent_emb
            )
            y_pred_i = clf.predict(X_i)[0]
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

    return y_pred_seq_test, info_parent_feature_test

# =====================
# Métricas / Relatórios
# =====================

def relatorio_geral(y_true, y_pred, titulo="Relatório"):
    """Calcula métricas simples: Accuracy e F1-macro (com zero_division=0)."""
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    rep = classification_report(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "classification_report": rep}

def print_resumo(rel_base_geral, rel_base_samecov_geral, rel_seq_geral,
                 total_pred, total_all, pct, threads_outros, origem_counts):
    """Imprime um resumo enxuto para notebooks (sem salvar arquivos)."""
    print("\n=== Resumo ===")
    print(f"Base (geral):      Acc={rel_base_geral['accuracy']:.3f} | F1={rel_base_geral['f1_macro']:.3f}")
    if not np.isnan(rel_base_samecov_geral['accuracy']):
        print(f"Base (cobert. SEQ): Acc={rel_base_samecov_geral['accuracy']:.3f} | F1={rel_base_samecov_geral['f1_macro']:.3f}")
    if not np.isnan(rel_seq_geral['accuracy']):
        print(f"Sequencial:        Acc={rel_seq_geral['accuracy']:.3f} | F1={rel_seq_geral['f1_macro']:.3f}")
    print(f"Cobertura SEQ:     {total_pred}/{total_all} ({pct:.1f}%)")
    if len(threads_outros) > 0:
        print("Threads com graus != {0,1}:", ", ".join(str(t) for t in threads_outros))
    if origem_counts:
        ordem = ["skipped_degree0", "assumed_concorda_degree1", "skipped_missing_parent_prediction", "predicted_from_test_parent"]
        print("Origem 'parent_label' no teste:")
        for k in ordem:
            if k in origem_counts:
                print(f" - {k}: {origem_counts[k]}")

# (removido) relatorio_por_grau: mantemos o relatório enxuto apenas no geral

# =====================
# MAIN
# =====================

def rodar_modelo_sequencial(
    dados_treino,
    dados_teste,
    n_estimators=200,
    random_state=42,
    usar_parent_emb=False,
    imprimir=True,
):
    """Roda a modelagem com UM modelo e DOIS protocolos de teste (Base x Sequencial).

    Parâmetros:
    - dados_treino/dados_teste: dicionários com chaves 'emb', 'target_emb', 'parent_label', 'label',
      'grau_distancia', opcionalmente 'parent_emb', e metadados como 'id', 'parent_id', 'thread'.
    - usar_parent_emb: se True, concatena também 'parent_emb' nas features.
    - imprimir: se True, imprime um resumo enxuto (recomendado em notebook).
    """
    # 1) Treinar UM único classificador no TREINO
    clf = treinar_modelo_base(
        dados_treino,
        n_estimators=n_estimators,
        random_state=random_state,
        usar_parent_emb=usar_parent_emb,
    )

    # 2) Preparar features do TESTE para o protocolo Base, aplicando a Regra II em grau=1
    graus_test = sanitize_grau(dados_teste['grau_distancia'])
    parent_label_base = ensure_2d_col(dados_teste['parent_label']).astype(float).copy()
    mask_grau1 = (graus_test == 1)
    if mask_grau1.any():
        parent_label_base[mask_grau1, 0] = 1.0  # força "Concorda" para grau 1

    X_te_base = concat_features(
        dados_teste['emb'],
        dados_teste['target_emb'],
        parent_label_base,
        dados_teste.get('parent_emb', None) if usar_parent_emb else None,
        usar_parent_emb=usar_parent_emb,
    )
    y_test_base = clf.predict(X_te_base)

    # 3) Rodar o protocolo Sequencial no TESTE (encadeado, sem rótulo real)
    y_pred_seq_test, parent_diag = inferir_sequencial_ids(
        clf, dados_teste, usar_parent_emb=usar_parent_emb
    )

    # 4) Métricas gerais e cobertura
    y_true_test = dados_teste['label']
    mask_seq_valid = (y_pred_seq_test != -1)  # itens realmente previstos no sequencial

    rel_base_geral = relatorio_geral(y_true_test, y_test_base)
    if mask_seq_valid.any():
        rel_base_samecov_geral = relatorio_geral(y_true_test[mask_seq_valid], y_test_base[mask_seq_valid])
        rel_seq_geral = relatorio_geral(y_true_test[mask_seq_valid], y_pred_seq_test[mask_seq_valid])
    else:
        rel_base_samecov_geral = {"accuracy": float('nan'), "f1_macro": float('nan'), "classification_report": ""}
        rel_seq_geral = {"accuracy": float('nan'), "f1_macro": float('nan'), "classification_report": ""}

    total_pred = int(mask_seq_valid.sum())
    total_all = int(len(y_true_test))
    pct = 100.0 * total_pred / max(total_all, 1)

    # 5) Diagnóstico simples de threads com graus fora do esperado (Exceção III)
    try:
        thr_list = np.asarray(dados_teste['thread'])
        mask_outros = ~np.isin(graus_test, [0, 1])
        threads_outros = sorted(set(thr_list[mask_outros].tolist()))
    except Exception:
        threads_outros = []

    # 6) Contagem simples da origem da feature do pai no teste (para auditoria)
    origem_counts = defaultdict(int)
    for d in parent_diag:
        origem_counts[d["parent_feature_source"]] += 1

    # 7) Impressão opcional do resumo
    if imprimir:
        print_resumo(
            rel_base_geral,
            rel_base_samecov_geral,
            rel_seq_geral,
            total_pred,
            total_all,
            pct,
            threads_outros,
            origem_counts,
        )

    # 8) Retorna resultados úteis para exploração no notebook
    return {
        "y_test_true": y_true_test,
        "y_test_base": y_test_base,
        "y_test_sequencial": y_pred_seq_test,
        "mask_seq_valid": mask_seq_valid,
        "rel_base_geral": rel_base_geral,
        "rel_base_samecov_geral": rel_base_samecov_geral,
        "rel_seq_geral": rel_seq_geral,
        "threads_outros": threads_outros,
        "origem_counts": dict(origem_counts),
        "params": {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "usar_parent_emb": usar_parent_emb,
        },
    }

# Exemplo de uso em notebook:
# dados_treino = joblib.load('input/input_conjuntoDeDados_treinamento.joblib')
# dados_teste  = joblib.load('input/input_conjuntoDeDados_teste.joblib')
# resultados = rodar_modelo_sequencial(dados_treino, dados_teste, usar_parent_emb=False, imprimir=True)
