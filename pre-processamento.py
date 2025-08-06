
# %% 
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')  # Verifica se está usando GPU ou CPU

# %%
import pandas as pd
import numpy as np
# import networkx as nx

# CARREGAR O DATASET
fileName = 'conjuntoDeDados_teste' # sem extensao

df_ori = pd.read_csv('Dados/'+ fileName + '.tsv', sep='\t', decimal = ',', encoding = 'UTF-8')
print(df_ori.columns)

# MANIPULAÇÃO DO DF

df = df_ori.copy()

# acessaremos as mensagens "parent" e "alvo/target" a partir do id. (Diminiuir o tanto de processamento p/ embeddings)
# vou criar linhas de mensagem Alvo

linhas_alvo = []
for thread, group in df.groupby('target_id'):
    msg = group['target_message'].iloc[0]

    linhas_alvo.append({
        'target_id': thread,
        'target_message': msg,
        'id': 'Alvo',
        'message': msg,
    })

df_tmp = pd.DataFrame(linhas_alvo)
df = pd.concat([df, df_tmp], ignore_index=True)



# %% -----------------------------------------
# GERAÇÃO DAS EMBEDDINGS

# https://huggingface.co/neuralmind/bert-base-portuguese-cased
# BERTimbau 
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads

# import torch
import joblib # salvar df de modo que mantenha os types

#Importando o Modelo BERTimbau
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = model.to(device) 
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Geração das Embeddings - função baseada no código dado pelo Prof.
def getEmbeddings(text, tokenizer, model):

    inputs = tokenizer(
        text, 
        return_tensors="pt",      
        truncation=True,          # corta textos longos para o tamanho máximo
        padding=True,             # adiciona padding para igualar o tamanho dos textos
        max_length=512            # tamanho máximo suportado pelo BERT
    )

    # Move todos os tensores do batch para a GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # função do torch que não utiliza gradientes (vamos apenas extrair embeddings)
    with torch.no_grad():
        outputs = model(**inputs)  # passa os tokens pelo modelo BERT
    # Pega o embedding do token [CLS] (primeiro token), que representa o texto inteiro
    return outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()

# %%
# SALVAR EMBEDDINGS

# Gerar embeddings das mensagens de cada comentário
embeddings = [getEmbeddings(text, tokenizer, model) for text in df['message']]

df['embedding'] = embeddings

# checar por duplicatas
df['embedding_tuple'] = df['embedding'].apply(lambda x: tuple(x)) # Converter cada embedding para tupla (ou string) para comparação
duplicates = df[df.duplicated(subset=['embedding_tuple'], keep=False)]  # Agora sim, checar por duplicatas de conteúdo
df = df.drop(columns=['embedding_tuple'])
# são mensagens duplicadass irrelevantes

# checar por nulos
nulls = df[df['embedding'].isnull()] # 0


# %% ---------------------------------------------
# Encoding das labels
# https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02/
# considerando "Discorda", "Neutro", "Concorda" como variáveis ordinais.
ordinalEncoding = {
    'Discorda': -1,
    'Neutro': 0,
    'Concorda': 1
}

# juntar labels "Neutro"
df['label'] = df['label'].replace('Discute', 'Neutro').replace('Pede Informações', 'Neutro').replace('Irrelevante', 'Neutro')
df['parent_label'] = df['parent_label'].replace('Discute', 'Neutro').replace('Pede Informações', 'Neutro').replace('Irrelevante', 'Neutro')

df['label_enc'] = df['label'].map(ordinalEncoding)
df['parent_label_enc'] = df['parent_label'].map(ordinalEncoding)

# %% --------------------------------------------------
# salvar o df
joblib.dump(df, 'embeddings/'+ fileName + '.joblib')
