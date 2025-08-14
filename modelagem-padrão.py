# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import joblib

fileName = 'conjuntoDeDados.joblib'
df_treino = joblib.load('embeddings/'+ fileName)

# %%

# Vamos utilizar 3 features: a embedding do comentário, a do alvo e a parent_label.

emb_atual = np.array(df_treino['embedding'].tolist())

# Embedding do Alvo
# Precisamos separar em dataframes por target_id
# Cria um dicionário: chave = target_id, valor = DataFrame da thread
threads = {target_id: group for target_id, group in df_treino.groupby('target_id')}

for target_id, dft in threads.items():
    # pegar embedding para cada id='Alvo'
    alvos = dft[(dft['id'] == 'Alvo') & (dft['target_id'] == target_id)]
    


X_combined = np.concatenate((emb_atual, parent_label, emb_alvo), axis=1)    # vetor de features

# %%
# https://gist.github.com/pb111/88545fa33780928694388779af23bf58
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
# Definindo o modelo de Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)



# %%
# Para realizarmos o k-fold, temos o problema de que a análise de cada comentário dependerá da thread que está inserido.
# Posso fazer de dois jeitos:
# 1- k-fold baseado pelas threads (target_id), ou seja, não separar a thread em folds diferentes
    # é mais interessante para o objetivo de futuramente utilizar predições de label dos pais
# 2- Adicionar colunas no df de embeddings do parent e do Alvo (cada linha terá 3 embeddings associadas)


