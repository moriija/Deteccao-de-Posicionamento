from sklearn.metrics import classification_report, accuracy_score, f1_score

# Validação cruzada e separação por grupos (thread)
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    GroupKFold,
    GroupShuffleSplit,  # separa treino/teste garantindo que grupos não se misturem
)
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

from modelagem import DataSet, FeatureBuilder, CVSelector
from modelagem import USE_CV  # flag para ativar/desativar CV

# =========== Estratégias de Validação Cruzada ===========
class CVSelector:
    """Seleciona o esquema de validação cruzada apropriado.

    Preferimos usar conhecimento de grupos (threads) para evitar que a mesma
    conversa apareça em treino e validação.
    """

    @staticmethod
    def make(y, groups):
        # Conta quantos grupos diferentes existem no treino
        if groups is not None:
            unique_groups = np.unique(groups)
            n_groups = unique_groups.shape[0]
        else:
            n_groups = 0

        # Se houver grupos suficientes, usa K-Fold com grupos
        if groups is not None and n_groups >= 2:
            n_splits = min(5, n_groups)
            if StratifiedGroupKFold is not None:
                return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                return GroupKFold(n_splits=n_splits)
        # Caso contrário, usa K-Fold estratificado comum (sem grupos)
        n_classes = np.unique(y).shape[0]
        n_splits = max(2, min(5, n_classes if n_classes >= 2 else 2))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# =========== Avaliador (CV + Teste + Relatórios) ===========
class ModelEvaluator:
    """Executa CV, treina no conjunto completo e gera relatórios.

    Métodos:
    - evaluate_cv_and_test: imprime métricas de CV e o relatório no teste.
    - evaluate_with_predicted_parent: avalia usando parent_label previsto
      (inclui grau=0) e detalha por grau.
    - compare_base_vs_sequential_by_depth: compara base vs. sequencial por grau (>0).
    """

    def __init__(self, estimator_factory, dataset: DataSet, cv):
        self.estimator_factory = estimator_factory  # função que cria um estimador
        self.ds = dataset
        self.cv = cv
        # Dicionário de métricas para o cross_validate
        self.scoring = {
            'acc': 'accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
        }
        # Fixamos as classes com base no treino para relatórios consistentes
        self.classes_ = np.unique(self.ds.y_train)

    def evaluate_cv_and_test(self, name: str, X_tr: np.ndarray, X_te: np.ndarray):
        """Roda CV no treino, treina final e imprime o relatório no teste."""
        print(f"\n{name}")

        # Cria um novo estimador (ex.: RandomForest)
        model = self.estimator_factory()

        # Validação cruzada (apenas no treino). Deixe USE_CV=False para pular.
        if USE_CV and self.cv is not None:
            resultados = cross_validate(
                model,
                X_tr,
                self.ds.y_train,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
                groups=self.ds.groups_train,
                return_train_score=False,
            )

            # Mostra a média e o desvio-padrão das métricas nos folds
            n_folds = getattr(self.cv, 'n_splits', 'n/a')
            print(
                f"CV ({n_folds}-fold) => "
                f"acc: {resultados['test_acc'].mean():.3f} +/- {resultados['test_acc'].std():.3f}, "
                f"f1_macro: {resultados['test_f1_macro'].mean():.3f} +/- {resultados['test_f1_macro'].std():.3f}, "
                f"f1_weighted: {resultados['test_f1_weighted'].mean():.3f} +/- {resultados['test_f1_weighted'].std():.3f}"
            )
        #else:
        #    print("CV: pulada (modo holdout 80/20)")

        # Treina no conjunto completo de treino
        model.fit(X_tr, self.ds.y_train)

        # Avalia no conjunto de teste
        y_pred = model.predict(X_te)
        print(classification_report(self.ds.y_test, y_pred, labels=self.classes_, zero_division=0))
        return model

    def evaluate_with_predicted_parent(self, trained_model):
        """Avalia o teste usando parent_label PREVISTO (inclui grau=0)."""
        # Monta X de teste a partir do parent_label previsto já injetado
        X_te_predpai = FeatureBuilder.dep_test_with_predicted_parent(self.ds)

        # Inclui grau==0 nas métricas (avaliamos todos os nós com grau definido)
        graus = self.ds.test.get('grau_distancia', None)
        mask = None
        if graus is not None and len(graus) == len(self.ds.y_test):
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None
            g_float = np.array([_to_float(v) for v in graus])
            # Considera todos os graus válidos (>= 0)
            mask = (g_float >= 0)

        if mask is not None:
            X_eval = X_te_predpai[mask]
            y_true = self.ds.y_test[mask]
            graus_eval = np.asarray(graus).ravel()[mask]
        else:
            X_eval = X_te_predpai
            y_true = self.ds.y_test
            graus_eval = np.asarray(graus).ravel() if graus is not None else None

        # Relatório agregado
        y_pred = trained_model.predict(X_eval)
        print("\nModelo Dependente: Com posicionamento pai previsto (inclui grau=0)")
        print(classification_report(y_true, y_pred, labels=self.classes_, zero_division=0))

        # Relatórios por grau de profundidade
        if graus_eval is None:
            print("Aviso: 'grau_distancia' ausente; sem análise por profundidade.")
            return

        valores = np.unique(graus_eval)
        """
        print("\nDesempenho por grau de profundidade:")
        for g in valores:
            idx = np.where(graus_eval == g)[0]
            if idx.size == 0:
                continue
            yt = y_true[idx]
            yp = y_pred[idx]
            acc = accuracy_score(yt, yp)
            f1m = f1_score(yt, yp, average='macro', zero_division=0)
            f1w = f1_score(yt, yp, average='weighted', zero_division=0)
            print(f"- grau={g} | n_test={idx.size} | acc={acc:.3f} | f1_macro={f1m:.3f} | f1_weighted={f1w:.3f}")
            print(classification_report(yt, yp, labels=self.classes_, zero_division=0)) """

    def compare_base_vs_sequential_by_depth(self, trained_model):
        """Compara métricas por grau entre BASE (pai real) e SEQUENCIAL (pai previsto).

        A comparação agora considera também o grau 0 (todos os graus válidos).
        """
        # Predições com parent_label REAL (base)
        _, X_te_base = FeatureBuilder.dep_true_parent_label(self.ds)
        y_pred_base = trained_model.predict(X_te_base)

        # Predições com parent_label PREVISTO (sequencial)
        X_te_seq = FeatureBuilder.dep_test_with_predicted_parent(self.ds)
        y_pred_seq = trained_model.predict(X_te_seq)

        # Máscara de graus válidos (>= 0)
        graus = self.ds.test.get('grau_distancia', None)
        if graus is None:
            print("Aviso: 'grau_distancia' ausente; sem comparação por profundidade.")
            return
        graus = np.asarray(graus).ravel()

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        g_float = np.array([_to_float(v) for v in graus])
        mask_valid = g_float >= 0

        # Lista de graus válidos para comparar
        graus_validos = np.unique(graus[mask_valid])
        if graus_validos.size == 0:
            print("Nenhum grau > 0 para comparar.")
            return

        base_rows, seq_rows = [], []    # dfs para os graficos de comparação
        print("\nComparação base vs. sequencial por grau (acc, f1_macro, f1_weighted):")
        for g in graus_validos:
            idx = (graus == g)  # subconjunto de linhas do teste para este grau
            yt = self.ds.y_test[idx]

            # BASE
            yp_base = y_pred_base[idx]
            acc_b = accuracy_score(yt, yp_base)
            f1m_b = f1_score(yt, yp_base, average='macro', zero_division=0)
            f1w_b = f1_score(yt, yp_base, average='weighted', zero_division=0)
            base_rows.append({'grau': g, 'n': int(idx.sum()), 'acc': acc_b, 'f1_macro': f1m_b, 'f1_weighted': f1w_b})
            
            # SEQUENCIAL
            yp_seq = y_pred_seq[idx]
            acc_s = accuracy_score(yt, yp_seq)
            f1m_s = f1_score(yt, yp_seq, average='macro', zero_division=0)
            f1w_s = f1_score(yt, yp_seq, average='weighted', zero_division=0)
            seq_rows.append({'grau': g, 'n': int(idx.sum()), 'acc': acc_s, 'f1_macro': f1m_s, 'f1_weighted': f1w_s})

            print(
                f"- grau={g} | n={idx.sum()} | "
                f"base: acc={acc_b:.3f}, f1_macro={f1m_b:.3f}, f1_weighted={f1w_b:.3f} | "
                f"sequencial: acc={acc_s:.3f}, f1_macro={f1m_s:.3f}, f1_weighted={f1w_s:.3f}"
            )

        return base_rows, seq_rows