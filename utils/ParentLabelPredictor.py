from modelagem import DataSet
import numpy as np

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

        # Itera os graus em ordem crescente
        degs = np.unique(g_float[np.isfinite(g_float)])
        degs = [d for d in degs]
        for d in sorted(degs):
            idxs = np.where(g_float == d)[0]
            for i in idxs:
                pid_key = self._norm_id(parent_ids_test[i])

                # 1) Determina o parent_label a ser usado como feature do filho i
                if d == 0:
                    p_lbl = 1  # regra: grau 0 sempre parent_label=1 (Concorda)
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

        # Cria a máscara de avaliação: consideramos todas as amostras com grau >= 0
        mask_valid = g_float >= 0
        self.ds.test['dep_eval_mask'] = mask_valid

        # Injeta predicted_p_label (Nx1) no dicionário de teste.
        # - Para grau==0, será 1 (pela regra acima).
        # - Para grau>0, será a label PREVISTA do pai.
        #   como maioria para manter o tipo de dado (dtype) estável.
        predicted_p = np.where(np.isfinite(g_float), predicted_parent_for_row, majority_label).astype(y_dtype)
        self.ds.test['predicted_p_label'] = predicted_p.reshape(-1, 1)
        print(f"predicted_p_label gerado em dados_teste: shape={self.ds.test['predicted_p_label'].shape}")