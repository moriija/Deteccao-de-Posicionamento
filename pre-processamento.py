# %% 
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')  # Verifica se está usando GPU ou CPU

# CARREGAR O DATASET
def carregar_dataset(fileName):
    # ...existing code...
    df_ori = pd.read_csv('Dados/'+ fileName + '.tsv', sep='\t', decimal = ',', encoding = 'UTF-8')
    print(df_ori.columns)
    return df_ori

# MANIPULAÇÃO DO DF
def manipular_dataframe(df_ori):
    # ...existing code...
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
    return df

#Importando o Modelo BERTimbau
def carregar_modelo_tokenizer():
    # https://huggingface.co/neuralmind/bert-base-portuguese-cased
    # BERTimbau 
    model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = model.to(device) 
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    return model, tokenizer

# Geração das Embeddings - função baseada no código dado pelo Prof.
def getEmbeddings(text, tokenizer, model):
    # ...existing code...
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

# SALVAR EMBEDDINGS
def gerar_embeddings(df, tokenizer, model):
    # Gerar embeddings das mensagens de cada comentário
    embeddings = [getEmbeddings(text, tokenizer, model) for text in df['message']]
    df['embedding'] = embeddings
    # checar por duplicatas
    df['embedding_tuple'] = df['embedding'].apply(lambda x: tuple(x)) # Converter cada embedding para tupla (ou string) para comparação
    duplicates = df[df.duplicated(subset=['embedding_tuple'], keep=False)]  # Agora sim, checar por duplicatas de conteúdo
    df = df.drop(columns=['embedding_tuple'])
    # checar por nulos
    nulls = df[df['embedding'].isnull()] # 0
    return df

# Encoding das labels
def encode_labels(df):
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
    return df
    

# Selecionando as Features
def selecionar_features(df):
    # Features: a embedding do comentário, a do alvo e a parent_label.
    emb_atual = np.array(df['embedding'].tolist())

    # Embedding do Alvo
    # vou separar em dataframes por target_id (cada thread). Deixará depois os diferentes jeitos de separação de features mais facil
    # Cria um dicionário: chave = target_id, valor = DataFrame da thread
    threads = {target_id: group for target_id, group in df.groupby('target_id')}

    # mapear msg alvo (id='Alvo' definido no pre-process) pra cada thread (target_id)
    target_id_to_emb_alvo = {}
    for target_id, thread_df in threads.items():
        target_id_to_emb_alvo[target_id] = thread_df[thread_df['id'] == 'Alvo']['embedding'].values[0]

    # lista das emb_alvo alinhada com as linhas do df confomre
    emb_alvos = []
    for target_id in df['target_id']:
        emb_alvos.append(target_id_to_emb_alvo[target_id])
    emb_alvos = np.array(emb_alvos)

    # precisamos que tenham o mesmo número de linhas (e dimnesões compatíveis)
    parent_label = df['parent_label_enc'].values.reshape(-1, 1) # Alterna estrutura pra 2D

    # FEATURES
    X_combined = np.concatenate((emb_alvos, emb_atual, parent_label), axis=1)
    """     
    # Verificando as dimensões
    print("Shape of emb_alvos:", emb_alvos.shape)
    print("Shape of parent_label:", parent_label.shape)
    print("Shape of emb_atual:", emb_atual.shape)
    print("Shape of X_combined:", X_combined.shape) """
    return X_combined


def processar_dataset(fileName):
    df_ori = carregar_dataset(fileName)
    df = manipular_dataframe(df_ori)
    model, tokenizer = carregar_modelo_tokenizer()
    df = gerar_embeddings(df, tokenizer, model)
    df = encode_labels(df)

    joblib.dump(df, 'embeddings/'+ fileName + '.joblib') # salvar o df com todas as transformações (p consulta)
    return df

# %% MAIN EXECUTION

def main():
    fileName = 'conjuntoDeDados'
    df = processar_dataset(fileName)
    X_combined = selecionar_features(df)

    dados_input = {
        'features': X_combined,
        'target': df['label_enc'].values,  # Labels codificadas
    }

    joblib.dump(dados_input, 'input/input_' + fileName + '.joblib')  # Salvar as features combinadas


if __name__ == "__main__":
    main()