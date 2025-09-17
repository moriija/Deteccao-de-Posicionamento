"""
O código está organizado em classes para facilitar a extensão:
- DataSet: empacota os dicionários de treino e teste.
- FeatureBuilder: monta as matrizes de features (X) para cada cenário.
- CVSelector: escolhe a validação cruzada adequada (com grupos por thread).
- ModelEvaluator: roda CV, avaliação no teste e relatórios por grau.
- ParentLabelPredictor: gera parent_label previsto no teste por profundidade.
"""

from utils.ParentLabelPredictor import ParentLabelPredictor
from utils.ModelEvaluator import ModelEvaluator
from utils.ModelEvaluator import CVSelector

import joblib  
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

# ===== Toggle para K-Fold =====
# Defina para False para pular a validação cruzada (usar apenas holdout 80/20)
# Ou simplesmente comente o bloco de CV em evaluate_cv_and_test abaixo.
USE_CV = False


# =========== Estruturas de dados ===========
class DataSet:
    # Container para manter os dados de treino e de teste.
    """
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
        """Vetor de labels do conjunto de treino (array 1D)."""
        return self.train['label']

    @property
    def y_test(self):
        """Vetor de labels do conjunto de teste (array 1D)."""
        return self.test['label']

    @property
    def groups_train(self):
        """Grupos (threads) do treino, usados na CV por grupo."""
        return self.train.get('thread', None)


# =========== Construtor de features (X) ===========
class FeatureBuilder:
    """Monta as matrizes de features (X) a partir dos dicionários.
    
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

    # Definir o esquema de validação cruzada (por grupo quando possível)
    cv = CVSelector.make(ds.y_train, ds.groups_train)

    # estimadores
    estimator_factory = lambda: RandomForestClassifier(n_estimators=100, random_state=42)

    evaluator = ModelEvaluator(estimator_factory, ds, cv)

    # Modelo Independente: Nenhum contexto estrutural (emb + target_emb)
    X_tr, X_te = FeatureBuilder.indep_no_context(ds)
    evaluator.evaluate_cv_and_test("Modelo Independente: Nenhum contexto estrutural", X_tr, X_te)

    # Modelo Independente: Com mensagem pai (emb + target_emb + parent_emb)
    X_tr, X_te = FeatureBuilder.indep_with_parent_emb(ds)
    evaluator.evaluate_cv_and_test("Modelo Independente: Com mensagem pai", X_tr, X_te)

    # Modelo Dependente: cenário BASE (usa parent_label real no teste)
    X_tr, X_te = FeatureBuilder.dep_true_parent_label(ds)
    modelo_dep = evaluator.evaluate_cv_and_test("Modelo Dependente: Com posicionamento pai", X_tr, X_te)


    # cenário SEQUENCIAL
    # Geração do parent_label previsto no teste (respeita profundidade)
    predictor = ParentLabelPredictor(ds)
    predictor.inject_predicted_parent_labels(modelo_dep)  # usa o MESMO modelo

    # Avaliação SEQUENCIAL (exclui grau=0) e comparação por grau
    evaluator.evaluate_with_predicted_parent(modelo_dep)
    base_rows, seq_rows = evaluator.compare_base_vs_sequential_by_depth(modelo_dep) # retorna dfs para graficos de avaliação
    
    # salvar os dados para os gráficos
    if base_rows and seq_rows:
        import pandas as pd
        df_base = pd.DataFrame(base_rows)
        df_seq = pd.DataFrame(seq_rows)
        df_base.to_csv('resultados\\modelagem_base_por_grau.csv', index=False)
        df_seq.to_csv('resultados\\modelagem_sequencial_por_grau.csv', index=False)
    

if __name__ == '__main__':
    main()

