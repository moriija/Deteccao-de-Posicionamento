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
        f"acc: {resultados['test_acc'].mean():.3f} ± {resultados['test_acc'].std():.3f}, "
        f"f1_macro: {resultados['test_f1_macro'].mean():.3f} ± {resultados['test_f1_macro'].std():.3f}, "
        f"f1_weighted: {resultados['test_f1_weighted'].mean():.3f} ± {resultados['test_f1_weighted'].std():.3f}"
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

