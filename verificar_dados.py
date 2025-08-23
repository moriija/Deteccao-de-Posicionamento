import pandas as pd
import numpy as np
import joblib

def verificar_estrutura_dados():
    """
    Verifica a estrutura dos dados para entender como mapear os graus de distância
    """
    print("VERIFICANDO ESTRUTURA DOS DADOS")
    print("="*50)
    
    # Carregar dados
    dados_treino = joblib.load('input/input_conjuntoDeDados_treinamento.joblib')
    dados_teste = joblib.load('input/input_conjuntoDeDados_teste.joblib')
    df_original = pd.read_csv('Dados/conjuntoDeDados_com_grau_distancia.tsv', sep='\t', encoding='UTF-8')
    
    print(f"Dados de treinamento: {len(dados_treino['features'])} amostras")
    print(f"Dados de teste: {len(dados_teste['features'])} amostras")
    print(f"Dados originais: {len(df_original)} linhas")
    
    print(f"\nColunas dos dados originais:")
    print(df_original.columns.tolist())
    
    print(f"\nColunas dos dados de treinamento:")
    print(list(dados_treino.keys()))
    
    print(f"\nColunas dos dados de teste:")
    print(list(dados_teste.keys()))
    
    # Verificar se há IDs nos dados
    if 'ids' in dados_treino:
        print(f"\nIDs de treinamento: {len(dados_treino['ids'])}")
        print(f"Primeiros 5 IDs: {dados_treino['ids'][:5]}")
    else:
        print("\n❌ Não há coluna 'ids' nos dados de treinamento")
    
    if 'ids' in dados_teste:
        print(f"IDs de teste: {len(dados_teste['ids'])}")
        print(f"Primeiros 5 IDs: {dados_teste['ids'][:5]}")
    else:
        print("❌ Não há coluna 'ids' nos dados de teste")
    
    # Verificar graus de distância
    print(f"\nGraus de distância únicos:")
    graus_unicos = df_original['grau_distancia'].value_counts().sort_index()
    print(graus_unicos)
    
    # Verificar se há IDs no df original
    print(f"\nIDs únicos no df original: {df_original['id'].nunique()}")
    print(f"Primeiros 5 IDs: {df_original['id'].head().tolist()}")
    
    # Verificar se há target_id
    if 'target_id' in df_original.columns:
        print(f"Target IDs únicos: {df_original['target_id'].nunique()}")
        print(f"Primeiros 5 target_ids: {df_original['target_id'].head().tolist()}")
    
    # Verificar se há parent_id
    if 'parent_id' in df_original.columns:
        print(f"Parent IDs únicos: {df_original['parent_id'].nunique()}")
        print(f"Primeiros 5 parent_ids: {df_original['parent_id'].head().tolist()}")
    
    return dados_treino, dados_teste, df_original

def criar_mapeamento_graus(dados_treino, dados_teste, df_original):
    """
    Cria um mapeamento entre os dados e os graus de distância
    """
    print(f"\nCRIANDO MAPEAMENTO DE GRAUS")
    print("="*50)
    
    # Como não temos IDs explícitos, vamos usar uma abordagem baseada na ordem
    # Assumir que os dados estão na mesma ordem que o df original
    
    # Filtrar apenas comentários (excluir linhas 'Alvo')
    df_comentarios = df_original[df_original['id'] != 'Alvo'].copy()
    df_comentarios = df_comentarios.reset_index(drop=True)
    
    print(f"Comentários válidos: {len(df_comentarios)}")
    print(f"Graus disponíveis: {sorted(df_comentarios['grau_distancia'].unique())}")
    
    # Verificar se o número de comentários corresponde aos dados de treinamento + teste
    total_dados = len(dados_treino['features']) + len(dados_teste['features'])
    print(f"Total de dados (treino + teste): {total_dados}")
    print(f"Total de comentários válidos: {len(df_comentarios)}")
    
    if len(df_comentarios) >= total_dados:
        print("✅ Número de comentários é suficiente para mapear todos os dados")
        
        # Criar mapeamento
        graus_treino = df_comentarios['grau_distancia'].iloc[:len(dados_treino['features'])].values
        graus_teste = df_comentarios['grau_distancia'].iloc[len(dados_treino['features']):len(dados_treino['features'])+len(dados_teste['features'])].values
        
        print(f"Graus de treinamento: {len(graus_treino)}")
        print(f"Distribuição: {np.bincount(graus_treino.astype(int))}")
        
        print(f"Graus de teste: {len(graus_teste)}")
        print(f"Distribuição: {np.bincount(graus_teste.astype(int))}")
        
        return graus_treino, graus_teste, df_comentarios
    else:
        print("❌ Número de comentários insuficiente")
        return None, None, None

if __name__ == "__main__":
    dados_treino, dados_teste, df_original = verificar_estrutura_dados()
    graus_treino, graus_teste, df_comentarios = criar_mapeamento_graus(dados_treino, dados_teste, df_original)
