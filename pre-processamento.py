# %% 
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import joblib
from calcular_grau_distancia import DepthCalculator

class ProcessadorDataset:
    def __init__(self, df_ori, device=None):
        self.df_ori = df_ori
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')  # Verifica se está usando GPU ou CPU
        self.df = df_ori.copy()
        self.model = None
        self.tokenizer = None

    # MANIPULAÇÃO DO DF
    def gerar_entrada_alvo(self):
        df = self.df
        # acessaremos as mensagens "parent" e "alvo/target" a partir do id. (Diminiuir o tanto de processamento p/ embeddings)
        # vou criar linhas de mensagem Alvo, para que possamos referencia-las conforme a thread
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
        self.df = df

    #Importando o Modelo BERTimbau
    def carregar_modelo_tokenizer(self):
        # https://huggingface.co/neuralmind/bert-base-portuguese-cased
        # BERTimbau 
        self.model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(self.device) 
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    

    # Geração das Embeddings - função baseada no código dado pelo Prof.
    def get_embeddings(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",      
            truncation=True,          # corta textos longos para o tamanho máximo
            padding=True,             # adiciona padding para igualar o tamanho dos textos
            max_length=512            # tamanho máximo suportado pelo BERT
        )
        # Move todos os tensores do batch para a GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # função do torch que não utiliza gradientes (vamos apenas extrair embeddings)
        with torch.no_grad():
            outputs = self.model(**inputs)  # passa os tokens pelo modelo BERT
        # Pega o embedding do token [CLS] (primeiro token), que representa o texto inteiro
        return outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()

    # SALVAR EMBEDDINGS
    def gerar_embeddings(self):
        # Gerar embeddings das mensagens de cada comentário
        df = self.df
        embeddings = [self.get_embeddings(text) for text in df['message']]
        df['embedding'] = embeddings
        # checar por duplicatas
        df['embedding_tuple'] = df['embedding'].apply(lambda x: tuple(x)) # Converter cada embedding para tupla (ou string) para comparação
        # duplicates = df[df.duplicated(subset=['embedding_tuple'], keep=False)]  # Agora sim, checar por duplicatas de conteúdo
        df = df.drop(columns=['embedding_tuple'])
        # checar por nulos
        # nulls = df[df['embedding'].isnull()] # 0
        self.df = df

    # Encoding das labels
    def encode_labels(self):
        # https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02/
        # considerando "Discorda", "Neutro", "Concorda" como variáveis ordinais.
        ordinal_encoding = {
            'Discorda': -1,
            'Neutro': 0,
            'Concorda': 1
        }
        df = self.df
        # Consolidar labels "Neutro"
        df['label'] = df['label'].replace({
            'Discute': 'Neutro', 
            'Pede Informações': 'Neutro', 
            'Irrelevante': 'Neutro'
        })
        df['parent_label'] = df['parent_label'].replace({
            'Discute': 'Neutro', 
            'Pede Informações': 'Neutro', 
            'Irrelevante': 'Neutro'
        })
        
        # Aplicar codificação
        df['label_enc'] = df['label'].map(ordinal_encoding)
        df['parent_label_enc'] = df['parent_label'].map(ordinal_encoding)
        
        self.df = df

    # Alterações no dataset e geração das embeddings (se precisa), salva como .joblib
    def processar_dataset(self, fileName):
        # manipulacao do dataset original 
        self.gerar_entrada_alvo()

        # geração das embeddings
        self.carregar_modelo_tokenizer()
        self.encode_labels()
        # geração do grau de distância
        calculator = DepthCalculator(self.df)
        self.df = calculator.processar_arquivo()

        self.gerar_embeddings()

        joblib.dump(self.df, 'embeddings/'+ fileName + '.joblib') # salvar o df com todas as transformações (p consulta)


    def gerar_input(self):
        df = self.df
        X_combined = self.selecionar_features()   # tambem filtra df
        dados_input = {
            'features': X_combined,
            'target': df['label_enc'].values,  # Labels codificadas
            'parent_label': df['parent_label_enc'].values.reshape(-1, 1) # Alterna estrutura pra 2D
        }
        return dados_input


    def selecionar_features(self):
        df = self.df
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

        # Filtrar emb_alvos para corresponder ao df filtrado
        mask = df['label_enc'].notna().values
        emb_alvos = emb_alvos[mask]

        # Precisa ser feito depois dos outros processamentos
        # Garante alinhamento com a lista de emb_alvos
        df_filtrado = df[mask].copy()

        # Features: a embedding do comentário, a do alvo e a parent_label.
        emb_atual = np.array(df_filtrado['embedding'].tolist())
        X_combined = np.concatenate((emb_alvos, emb_atual), axis=1)

        """     
        # Verificando as dimensões (DEBUG)
        print("Shape of emb_alvos:", emb_alvos.shape)
        print("Shape of parent_label:", parent_label.shape)
        print("Shape of emb_atual:", emb_atual.shape)
        print("Shape of X_combined:", X_combined.shape) """
        self.df = df_filtrado
        return X_combined

# --------------------------------------------------

def main():
    fileNames = ['conjuntoDeDados', 'conjuntoDeDados_teste', 'conjuntoDeDados_treinamento']
    # df = joblib.load('embeddings/' + fileName + '.joblib')  # df com embeddings e labels codificadas

    for fileName in fileNames:
        print(f'Processando {fileName}...')
        df = pd.read_csv('Dados/'+ fileName + '.tsv', sep='\t', decimal = ',', encoding = 'UTF-8')

        processador = ProcessadorDataset(df)
        processador.processar_dataset(fileName)
        dados_input = processador.gerar_input()

        joblib.dump(dados_input, 'input/input_' + fileName + '.joblib')  # Salvar as features combinadas


if __name__ == "__main__":
    main()