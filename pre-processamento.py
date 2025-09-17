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

        df['embedding'] = [self.get_embeddings(text) for text in df['message']]
        df['parent_emb'] = [self.get_embeddings(text) for text in df['parent_message']]
        df['target_emb'] = [self.get_embeddings(text) for text in df['target_message']]

        # checar por duplicatas
        # df['embedding_tuple'] = df['embedding'].apply(lambda x: tuple(x)) # Converter cada embedding para tupla (ou string) para comparação
        # duplicates = df[df.duplicated(subset=['embedding_tuple'], keep=False)]  # Agora sim, checar por duplicatas de conteúdo
        # df = df.drop(columns=['embedding_tuple'])
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

        # Precisa ser feito depois dos outros processamentos
        # Garante alinhamento perturbado por linhas não convenientes como Alvos
        mask = (
            self.df['label_enc'].notna() &
            self.df['message'].notna() &
            self.df['parent_message'].notna() &
            self.df['target_message'].notna()
        )
        self.df = self.df[mask].copy()
        self.gerar_embeddings()


    def selecionar_features(self):
        df = self.df

        # Features: a embedding do comentário, a do alvo e a parent_label.
        emb_atual = np.array(df['embedding'].tolist())
        emb_alvo = np.array(df['target_emb'].tolist())
        emb_parent = np.array(df['parent_emb'].tolist())

        """     
        # Verificando as dimensões (DEBUG)
        print("Shape of emb_alvo:", emb_alvos.shape)
        """
        graus_distancia = df['grau_distancia'].values  # Array alinhado dos graus de distância

        dados_input = {
            'id': df['id'].values,  # IDs originais
            'parent_id': df['parent_id'].values,  # IDs dos pais
            'target_emb': emb_alvo,
            'emb': emb_atual,
            'parent_emb': emb_parent,
            'label': df['label_enc'].values,  # Labels codificadas
            'parent_label': df['parent_label_enc'].values.reshape(-1, 1), # formato 2D, necessario pra modelagem
            'thread': df['target_id'].values, # ID da thread
            'grau_distancia': graus_distancia  # Array alinhado dos graus de distância
        }
        return dados_input, df

# --------------------------------------------------

def main():
    fileNames = [
        'conjuntoDeDados'
                 ]
    # df = joblib.load('embeddings/' + fileName + '.joblib')  # df com embeddings e labels codificadas

    for fileName in fileNames:
        print(f'\n ===== Processando {fileName}... =======')
        df = pd.read_csv('Dados/'+ fileName + '.tsv', sep='\t', decimal = ',', encoding = 'UTF-8')

        processador = ProcessadorDataset(df)
        processador.processar_dataset(fileName)
        dados_input, df_processado = processador.selecionar_features()

        joblib.dump(dados_input, 'input/input_' + fileName + '.joblib')  # Salvar as features combinadas
        joblib.dump(df_processado, 'Dados_preprocessados/'+ fileName + '.joblib') # salvar o df com todas as transformações (p consulta)


if __name__ == "__main__":
    main()