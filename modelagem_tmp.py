import joblib
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold, GroupKFold
from sklearn.model_selection import StratifiedGroupKFold


dados = joblib.load('input\\input_conjuntoDeDados_treinamento.joblib')
dados_teste = joblib.load('input\\input_conjuntoDeDados_teste.joblib')

# Targets e grupos para CV
y_train = dados['label']
y_test = dados_teste['label']
groups_train = dados.get('thread', None)


# Escolha de esquema de CV (group-aware por thread se disponível)
def _make_cv(y, groups):
    if groups is not None:
        unique_groups = numpy.unique(groups)
        n_groups = unique_groups.shape[0]
    else:
        n_groups = 0

    if groups is not None and n_groups >= 2:
        n_splits = min(5, n_groups)
        if StratifiedGroupKFold is not None:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            return GroupKFold(n_splits=n_splits)
    else:
        # Fallback: estratificado sem grupos
        n_classes = numpy.unique(y).shape[0]
        n_splits = max(2, min(5, n_classes if n_classes >= 2 else 2))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


cv = _make_cv(y_train, groups_train)


def avalia_modelo(nome, X_tr, X_te):
    print(f"\n{nome}")
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scoring = {
        'acc': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
    }
    resultados = cross_validate(
        modelo_rf,
        X_tr,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        groups=groups_train,
        return_train_score=False,
    )
    n_folds = getattr(cv, 'n_splits', 'n/a')
    print(
        f"CV ({n_folds}-fold) => "
        f"acc: {resultados['test_acc'].mean():.3f} +/- {resultados['test_acc'].std():.3f}, "
        f"f1_macro: {resultados['test_f1_macro'].mean():.3f} +/- {resultados['test_f1_macro'].std():.3f}, "
        f"f1_weighted: {resultados['test_f1_weighted'].mean():.3f} +/- {resultados['test_f1_weighted'].std():.3f}"
    )

    # Ajuste final no treino completo e avaliação no conjunto de teste
    modelo_rf.fit(X_tr, y_train)
    y_pred = modelo_rf.predict(X_te)
    print(classification_report(y_test, y_pred))


# Independente: sem contexto estrutural
features_tr = numpy.concatenate((dados['emb'], dados['target_emb']), axis=1)
features_te = numpy.concatenate((dados_teste['emb'], dados_teste['target_emb']), axis=1)
avalia_modelo("Modelo Independente: Nenhum contexto estrutural", features_tr, features_te)

# Independente: com contexto da mensagem pai
features_tr = numpy.concatenate((dados['emb'], dados['target_emb'], dados['parent_emb']), axis=1)
features_te = numpy.concatenate((dados_teste['emb'], dados_teste['target_emb'], dados_teste['parent_emb']), axis=1)
avalia_modelo("Modelo Independente: Com mensagem pai", features_tr, features_te)

# Dependente: com parent_label
features_tr = numpy.concatenate((dados['emb'], dados['target_emb'], dados['parent_label']), axis=1)
features_te = numpy.concatenate((dados_teste['emb'], dados_teste['target_emb'], dados_teste['parent_label']), axis=1)
avalia_modelo("Modelo Dependente: Com posicionamento pai", features_tr, features_te)


# ===== Geração de predicted_p_label em dados_teste =====
# Treina o mesmo modelo dependente no treino completo
X_dep_tr = numpy.concatenate((dados['emb'], dados['target_emb'], dados['parent_label']), axis=1)
X_dep_te = numpy.concatenate((dados_teste['emb'], dados_teste['target_emb'], dados_teste['parent_label']), axis=1)

modelo_dep = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_dep.fit(X_dep_tr, y_train)

# Prediz os rótulos para todas as linhas do teste (serão usados como rótulos dos pais)
y_pred_test_all = modelo_dep.predict(X_dep_te)


def _norm_id(v):
    """Normaliza ID para string; trata None/NaN retornando None."""
    try:
        if v is None:
            return None
        if isinstance(v, float) and numpy.isnan(v):
            return None
    except Exception:
        pass
    return str(v)


# Mapas id -> índice para localizar pais no conjunto de teste e/ou treino
ids_test = dados_teste['id']
ids_train = dados['id']
id_to_idx_test = {}
for i in range(len(ids_test)):
    key = _norm_id(ids_test[i])
    if key is not None:
        id_to_idx_test[key] = i
id_to_idx_train = {}
for i in range(len(ids_train)):
    key = _norm_id(ids_train[i])
    if key is not None:
        id_to_idx_train[key] = i

parent_ids_test = dados_teste['parent_id']

# Fallback: rótulo majoritário do treino, caso o pai não esteja em teste nem treino
values, counts = numpy.unique(y_train, return_counts=True)
majority_label = values[numpy.argmax(counts)]

predicted_p = numpy.empty(len(parent_ids_test), dtype=y_pred_test_all.dtype)

for i in range(len(parent_ids_test)):
    pid_key = _norm_id(parent_ids_test[i])
    if pid_key is not None and pid_key in id_to_idx_test:
        # Pai está no conjunto de teste: usa a predição feita para a linha do pai
        predicted_p[i] = y_pred_test_all[id_to_idx_test[pid_key]]
    elif pid_key is not None and pid_key in id_to_idx_train:
        # Pai está no treino: calcula predição específica do pai usando o modelo dependente
        pidx = id_to_idx_train[pid_key]
        X_p = numpy.concatenate(
            (
                dados['emb'][pidx].reshape(1, -1),
                dados['target_emb'][pidx].reshape(1, -1),
                dados['parent_label'][pidx].reshape(1, -1),
            ),
            axis=1,
        )
        predicted_p[i] = modelo_dep.predict(X_p)[0]
    else:
        # Pai não encontrado: aplica fallback
        predicted_p[i] = majority_label

# Adiciona as labels previstas dos pais!
dados_teste['predicted_p_label'] = predicted_p.reshape(-1, 1)
# print(f"predicted_p_label gerado em dados_teste: shape={dados_teste['predicted_p_label'].shape}")

