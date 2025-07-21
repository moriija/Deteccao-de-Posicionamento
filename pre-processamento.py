# %%
import pandas as pd
import networkx as nx
df_ori = pd.read_csv('Dados/aborto-consolidated-parent-based_treinamento.tsv', sep='\t', decimal = ',', encoding = 'UTF-8')
print(df_ori.columns)

# %%

# MANIPULAÇÃO DO DF

df = df_ori.copy()
# acessaremos as mensagens "parent" e "alvo/target" a partir do id. (Diminiuir o tanto de processamento p/ embeddings)
df = df.drop(columns=['parent_message', 'target_and_message', 'target_parent_message', 'target_message'])
# deixaremos parent_label por enquanto
# target_id é a chave da thread dos comentarios com o mesmo Alvo.

# %%
# definir a label do comentário original como "Comentário Original". (Apenas metade das linhas tinham)
df.loc[df['id'] == '1', 'label'] = 'Comentário Original'

# todos que tiverem original como parent terão parent_label = 'Comentário Original'
df.loc[df['parent_id'] == '1', 'parent_label'] = 'Comentário Original'

# podemos considerar "Comentário Original" posteriormente como "Concorda".

# %%