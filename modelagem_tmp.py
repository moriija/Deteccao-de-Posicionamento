"""
Modelagem sequencial de posicionamento (versão comentada para iniciantes)

Este script treina e avalia um classificador de posicionamento
(Discorda/Neutro/Concorda) levando em conta a estrutura de threads
(comentário-resposta). Ele também executa uma avaliação "sequencial":
ao prever a linha de um comentário, usa a label PREVISTA do seu
comentário-pai como entrada do modelo (em vez da label real) — isso
simula o uso em produção, onde não conhecemos as labels verdadeiras.

Etapas principais do fluxo:
1) Carrega um ÚNICO arquivo de dados pré-processados (features prontas).
2) Separa treino e teste por thread (para evitar vazamento de informação).
3) Treina um modelo dependente (usa emb + target_emb + parent_label).
4) Gera, no teste, as labels de pais de forma sequencial por profundidade.
   - Para grau=1, força parent_label=1 (Concorda) ao prever o próprio nó.
   - Para grau>1, usa a label PREVISTA do pai já calculada em graus menores.
5) Avalia desempenho com parent_label real (base) e com parent_label
   previsto (sequencial), incluindo métricas por grau de profundidade.

O código está organizado em classes para facilitar a extensão:
- DataSet: empacota os dicionários de treino e teste.
- FeatureBuilder: monta as matrizes de features (X) para cada cenário.
- CVSelector: escolhe a validação cruzada adequada (com grupos por thread).
- ModelEvaluator: roda CV, avaliação no teste e relatórios por grau.
- ParentLabelPredictor: gera parent_label previsto no teste por profundidade.
"""

# =========== Imports básicos ===========
import joblib  # ler/salvar objetos Python (dicionários com arrays, etc.)
import numpy as np  # operações numéricas e manipulação de arrays

# Modelo e métricas do scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Validação cruzada e separação por grupos (thread)
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    GroupKFold,
    GroupShuffleSplit,  # separa treino/teste garantindo que grupos não se misturem
)

from sklearn.model_selection import StratifiedGroupKFold

# ===== Toggle simples para K-Fold =====
# Defina para False para pular a validação cruzada (usar apenas holdout 80/20)
# Ou simplesmente comente o bloco de CV em evaluate_cv_and_test abaixo.
USE_CV = False



# =========== Estruturas de dados ===========
class DataSet:
    """Container simples para manter os dados de treino e de teste.

    Espera dicionários (train_dict e test_dict):
    - 'emb', 'target_emb', 'parent_emb' (arrays 2D: n_amostras x n_features)
    - 'parent_label' (array 2D: n_amostras x 1)
    - 'label' (array 1D com as classes -1, 0, 1)
    - 'thread' (ID da thread para agrupamento)
    - 'id', 'parent_id' e 'grau_distancia'
    """

    def __init__(self, train_dict: dict, test_dict: dict):
        self.train = train_dict
        self.test = test_dict

    @property
    def y_train(self):
        """Retorna o vetor de labels do conjunto de treino (array 1D)."""
        return self.train['label']

    @property
    def y_test(self):
        """Retorna o vetor de labels do conjunto de teste (array 1D)."""
        return self.test['label']

    @property
    def groups_train(self):
        """Retorna os grupos (threads) do treino, usados na CV por grupo."""
        return self.train.get('thread', None)


# =========== Construtor de features (X) ===========
class FeatureBuilder:
    """Monta as matrizes de features (X) a partir dos dicionários.

    Observações sobre formatos:
    - 'emb', 'target_emb' e 'parent_emb' são arrays 2D (n, d).
    - 'parent_label' e 'predicted_p_label' são arrays 2D de uma coluna (n, 1).
    - As features são concatenadas por coluna (axis=1).
    """

    @staticmethod
    def dep_true_parent_label(ds: 'DataSet'):
        """Cenário BASE (usa parent_label real do teste)."""
        X_tr = np.concatenate(
            (ds.train['emb'], ds.train['target_emb'], ds.train['parent_label']), axis=1
        )
        X_te = np.concatenate(
            (ds.test['emb'], ds.test['target_emb'], ds.test['parent_label']), axis=1
        )
        return X_tr, X_te

    @staticmethod
    def dep_test_with_predicted_parent(ds: 'DataSet'):
        """Cenário SEQUENCIAL (usa parent_label previsto no teste)."""
        X_te = np.concatenate(
            (ds.test['emb'], ds.test['target_emb'], ds.test['predicted_p_label']), axis=1
        )
        return X_te

    @staticmethod
    def indep_no_context(ds: 'DataSet'):
        """Cenário INDEPENDENTE (sem contexto estrutural): usa apenas emb + target_emb."""
        X_tr = np.concatenate((ds.train['emb'], ds.train['target_emb']), axis=1)
        X_te = np.concatenate((ds.test['emb'], ds.test['target_emb']), axis=1)
        return X_tr, X_te

    @staticmethod
    def indep_with_parent_emb(ds: 'DataSet'):
        """Cenário INDEPENDENTE (com mensagem pai): usa emb + target_emb + parent_emb."""
        X_tr = np.concatenate((ds.train['emb'], ds.train['target_emb'], ds.train['parent_emb']), axis=1)
        X_te = np.concatenate((ds.test['emb'], ds.test['target_emb'], ds.test['parent_emb']), axis=1)
        return X_tr, X_te


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
      (exclui grau=0) e detalha por grau.
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
        """Avalia o teste usando parent_label PREVISTO (exclui grau=0)."""
        # Monta X de teste a partir do parent_label previsto já injetado
        X_te_predpai = FeatureBuilder.dep_test_with_predicted_parent(self.ds)

        # Exclui grau==0 das métricas (avaliamos apenas nós com pai)
        graus = self.ds.test.get('grau_distancia', None)
        mask = None
        if graus is not None and len(graus) == len(self.ds.y_test):
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None
            g_float = np.array([_to_float(v) for v in graus])
            mask = (g_float > 0)

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
        print("\nModelo Dependente: Com posicionamento pai previsto (exclui grau=0)")
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

        A comparação considera apenas graus > 0 no cenário sequencial.
        """
        # Predições com parent_label REAL (base)
        _, X_te_base = FeatureBuilder.dep_true_parent_label(self.ds)
        y_pred_base = trained_model.predict(X_te_base)

        # Predições com parent_label PREVISTO (sequencial)
        X_te_seq = FeatureBuilder.dep_test_with_predicted_parent(self.ds)
        y_pred_seq = trained_model.predict(X_te_seq)

        # Máscara de graus > 0
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
        mask_gt0 = g_float > 0

        # Lista de graus válidos para comparar
        graus_validos = np.unique(graus[mask_gt0])
        if graus_validos.size == 0:
            print("Nenhum grau > 0 para comparar.")
            return

        print("\nComparação base vs. sequencial por grau (acc, f1_macro, f1_weighted):")
        for g in graus_validos:
            idx = (graus == g)  # subconjunto de linhas do teste para este grau
            yt = self.ds.y_test[idx]

            # BASE (parent_label real)
            yp_base = y_pred_base[idx]
            acc_b = accuracy_score(yt, yp_base)
            f1m_b = f1_score(yt, yp_base, average='macro', zero_division=0)
            f1w_b = f1_score(yt, yp_base, average='weighted', zero_division=0)

            # SEQUENCIAL (parent_label previsto já injetado)
            yp_seq = y_pred_seq[idx]
            acc_s = accuracy_score(yt, yp_seq)
            f1m_s = f1_score(yt, yp_seq, average='macro', zero_division=0)
            f1w_s = f1_score(yt, yp_seq, average='weighted', zero_division=0)

            print(
                f"- grau={g} | n={idx.sum()} | "
                f"base: acc={acc_b:.3f}, f1_macro={f1m_b:.3f}, f1_weighted={f1w_b:.3f} | "
                f"sequencial: acc={acc_s:.3f}, f1_macro={f1m_s:.3f}, f1_weighted={f1w_s:.3f}"
            )


# =========== Previsor sequencial de parent_label (teste) ===========
class ParentLabelPredictor:
    """Gera 'predicted_p_label' no teste de forma realmente sequencial por grau.

    Regras aplicadas:
    - grau=1: força parent_label=1 (Concorda) ao prever a PRÓPRIA linha.
    - grau>1: usa a label PREVISTA do pai (no próprio teste), prevista em uma
      profundidade anterior. Se o pai existir apenas no treino, prevemos a label
      do pai com o mesmo modelo (sem cadeia recursiva no treino). Se não existir,
      usamos a classe majoritária do treino como fallback.
    """

    def __init__(self, dataset: DataSet):
        self.ds = dataset

    @staticmethod
    def _norm_id(v):
        """Converte um ID em string e trata None/NaN retornando None."""
        try:
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
        except Exception:
            pass
        return str(v)

    def inject_predicted_parent_labels(self, trained_model):
        """Cria e injeta 'predicted_p_label' (Nx1) em self.ds.test.

        Também salva 'dep_eval_mask' (booleano), indicando quais entradas têm
        grau > 0 para serem usadas na avaliação sequencial.
        """
        # Número de linhas no teste
        N = len(self.ds.test['id'])
        # Garantimos que o dtype dos rótulos previstos siga o das labels de treino
        y_dtype = self.ds.y_train.dtype

        # Mapas de ID -> índice (para localizar pais no teste/treino)
        ids_test = self.ds.test['id']
        ids_train = self.ds.train['id']
        id_to_idx_test = {}
        for i in range(len(ids_test)):
            k = self._norm_id(ids_test[i])
            if k is not None:
                id_to_idx_test[k] = i
        id_to_idx_train = {}
        for i in range(len(ids_train)):
            k = self._norm_id(ids_train[i])
            if k is not None:
                id_to_idx_train[k] = i

        parent_ids_test = self.ds.test['parent_id']
        graus = self.ds.test.get('grau_distancia', None)
        if graus is None:
            raise ValueError("'grau_distancia' ausente em dados_teste — necessário para ordem sequencial.")

        # Converte graus em floats (seguro para strings/NaN)
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        g_float = np.array([_to_float(v) for v in graus])

        # Classe majoritária do treino (fallback quando não achamos pai)
        values, counts = np.unique(self.ds.y_train, return_counts=True)
        majority_label = values[np.argmax(counts)]

        # Buffers para armazenar:
        # - a label PREVISTA do pai (por linha de teste)
        # - a label PREVISTA da própria linha de teste
        predicted_parent_for_row = np.empty(N, dtype=y_dtype)
        predicted_label_test = np.empty(N, dtype=y_dtype)

        # Itera os graus em ordem crescente, ignorando grau 0
        degs = np.unique(g_float[np.isfinite(g_float)])
        degs = [d for d in degs if d > 0]
        for d in sorted(degs):
            idxs = np.where(g_float == d)[0]
            for i in idxs:
                pid_key = self._norm_id(parent_ids_test[i])

                # 1) Determina o parent_label a ser usado como feature do filho i
                if d == 1:
                    p_lbl = 1  # regra: grau 1 sempre parent_label=1 (Concorda)
                else:
                    if pid_key is not None and pid_key in id_to_idx_test:
                        # Pai está no teste e já foi previsto em um grau anterior
                        pidx = id_to_idx_test[pid_key]
                        p_lbl = predicted_label_test[pidx]
                    elif pid_key is not None and pid_key in id_to_idx_train:
                        # Pai só existe no treino: prevemos a label do PAI
                        tr_idx = id_to_idx_train[pid_key]
                        X_p = np.concatenate(
                            (
                                self.ds.train['emb'][tr_idx].reshape(1, -1),
                                self.ds.train['target_emb'][tr_idx].reshape(1, -1),
                                self.ds.train['parent_label'][tr_idx].reshape(1, -1),
                            ),
                            axis=1,
                        )
                        p_lbl = trained_model.predict(X_p)[0]
                    else:
                        # Não encontramos pai ⇒ usa maioria
                        p_lbl = majority_label

                predicted_parent_for_row[i] = p_lbl

                # 2) Prevejo a label da PRÓPRIA linha i (filho) usando o parent_label definido
                X_i = np.concatenate(
                    (
                        self.ds.test['emb'][i].reshape(1, -1),
                        self.ds.test['target_emb'][i].reshape(1, -1),
                        np.array([[p_lbl]]),
                    ),
                    axis=1,
                )
                predicted_label_test[i] = trained_model.predict(X_i)[0]

        # Cria a máscara de avaliação: consideramos apenas amostras com grau > 0
        mask_gt0 = g_float > 0
        self.ds.test['dep_eval_mask'] = mask_gt0

        # Injeta predicted_p_label (Nx1) no dicionário de teste.
        # - Para grau==1, será 1 (pela regra acima).
        # - Para grau>1, será a label PREVISTA do pai.
        # - Para grau==0, o valor não será usado (máscara exclui), mas definimos
        #   como maioria para manter o tipo de dado (dtype) estável.
        predicted_p = np.where(mask_gt0, predicted_parent_for_row, majority_label).astype(y_dtype)
        self.ds.test['predicted_p_label'] = predicted_p.reshape(-1, 1)
        print(f"predicted_p_label gerado em dados_teste: shape={self.ds.test['predicted_p_label'].shape}")


# =========== Ponto de entrada do script ===========
def main():
    """Carrega dados, treina e avalia o modelo nos cenários base e sequencial."""

    # 1) Carrega um ÚNICO dicionário com todas as features/labels
    all_data = joblib.load('input\\input_conjuntoDeDados.joblib')

    # 2) Separa treino e teste por thread (GroupShuffleSplit) para evitar
    #    que mensagens da mesma thread apareçam em treino e teste.
    def split_train_test_by_thread(d: dict, test_size=0.2, random_state=42):
        groups = np.asarray(d['thread']).ravel()
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        idx_train, idx_test = next(gss.split(np.zeros(len(groups)), groups=groups))
        # Seleciona os índices em cada chave do dicionário
        d_train = {k: (v[idx_train] if hasattr(v, '__getitem__') else v) for k, v in d.items()}
        d_test = {k: (v[idx_test] if hasattr(v, '__getitem__') else v) for k, v in d.items()}
        return d_train, d_test

    dados, dados_teste = split_train_test_by_thread(all_data)
    ds = DataSet(dados, dados_teste)

    # 3) Define o esquema de validação cruzada (por grupo quando possível)
    cv = CVSelector.make(ds.y_train, ds.groups_train)

    # 4) Fábrica de estimadores (fácil trocar por outro modelo/pipeline)
    estimator_factory = lambda: RandomForestClassifier(n_estimators=100, random_state=42)

    evaluator = ModelEvaluator(estimator_factory, ds, cv)

    # 4.1) Modelo Independente: Nenhum contexto estrutural (emb + target_emb)
    X_tr, X_te = FeatureBuilder.indep_no_context(ds)
    evaluator.evaluate_cv_and_test("Modelo Independente: Nenhum contexto estrutural", X_tr, X_te)

    # 4.2) Modelo Independente: Com mensagem pai (emb + target_emb + parent_emb)
    X_tr, X_te = FeatureBuilder.indep_with_parent_emb(ds)
    evaluator.evaluate_cv_and_test("Modelo Independente: Com mensagem pai", X_tr, X_te)

    # 5) Cenário BASE (usa parent_label real no teste)
    X_tr, X_te = FeatureBuilder.dep_true_parent_label(ds)
    modelo_dep = evaluator.evaluate_cv_and_test("Modelo Dependente: Com posicionamento pai", X_tr, X_te)

    # 6) Geração SEQUENCIAL do parent_label previsto no teste (respeita profundidade)
    predictor = ParentLabelPredictor(ds)
    predictor.inject_predicted_parent_labels(modelo_dep)  # usa o MESMO modelo

    # 7) Avaliação SEQUENCIAL (exclui grau=0) e comparação por grau
    evaluator.evaluate_with_predicted_parent(modelo_dep)
    evaluator.compare_base_vs_sequential_by_depth(modelo_dep)


if __name__ == '__main__':
    main()
