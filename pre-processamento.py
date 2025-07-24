# %%
import pandas as pd
import numpy as np
# import networkx as nx

# %% 

# https://huggingface.co/neuralmind/bert-base-portuguese-cased
# BERTimbau 
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

import torch
import joblib # salvar df de modo que mantenha os types

# %%
df_ori = pd.read_csv('Dados/aborto-consolidated-parent-based_treinamento.tsv', sep='\t', decimal = ',', encoding = 'UTF-8')
print(df_ori.columns)

# MANIPULAÇÃO DO DF

df = df_ori.copy()
# acessaremos as mensagens "parent" e "alvo/target" a partir do id. (Diminiuir o tanto de processamento p/ embeddings)
df = df.drop(columns=['parent_message', 'target_and_message', 'target_parent_message', 'target_message'])
# deixaremos parent_label por enquanto
# target_id é a chave da thread dos comentarios com o mesmo Alvo.

# definir a label do comentário original como "Comentário Original". (Apenas metade das linhas tinham)
df.loc[df['id'] == '1', 'label'] = 'Comentário Original'

# todos que tiverem original como parent terão parent_label = 'Comentário Original'
df.loc[df['parent_id'] == '1', 'parent_label'] = 'Comentário Original'

# podemos considerar "Comentário Original" posteriormente como "Concorda".

# %%
#Importando o Modelo BERTimbau
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")


# %%
# Geração das Embeddings - função baseada no código dado pelo Prof.
def getEmbeddings(text, tokenizer, model):

    inputs = tokenizer(
        text, 
        return_tensors="pt",      
        truncation=True,          # corta textos longos para o tamanho máximo
        padding=True,             # adiciona padding para igualar o tamanho dos textos
        max_length=512            # tamanho máximo suportado pelo BERT
    )

    # função do torch que não utiliza gradientes (vamos apenas extrair embeddings)
    with torch.no_grad():
        outputs = model(**inputs)  # passa os tokens pelo modelo BERT
    # Pega o embedding do token [CLS] (primeiro token), que representa o texto inteiro
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# %%
# Gerar embeddings das mensagens de cada comentário
embeddings = [getEmbeddings(text, tokenizer, model) for text in df['message']]

# %%
df['embedding'] = embeddings

# checar por duplicatas
df['embedding_tuple'] = df['embedding'].apply(lambda x: tuple(x)) # Converter cada embedding para tupla (ou string) para comparação
duplicates = df[df.duplicated(subset=['embedding_tuple'], keep=False)]  # Agora sim, checar por duplicatas de conteúdo
df = df.drop(columns=['embedding_tuple'])
# são mensagens duplicadass irrelevantes

# checar por nulos
nulls = df[df['embedding'].isnull()] # 0


# %%
# salvar o DataFrame com as embeddings (demora pra gerar )
joblib.dump(df, 'embeddings/aborto-consolidated-parent-based_treinamento-embeddings.joblib')
