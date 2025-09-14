import pandas as pd
import networkx as nx
import numpy as np

def calcular_grau_distancia_v2(df):
    """
    Versão melhorada que calcula o grau de distância de cada comentário em relação ao comentário original da thread.
    Lida melhor com casos onde o comentário original não está presente.
    
    Args:
        df: DataFrame com as colunas: id, parent_id, target_id, parent_label, parent_name
        
    Returns:
        DataFrame com nova coluna 'grau_distancia' adicionada
    """
    df_copy = df.copy()
    
    # Adicionar coluna de grau de distância
    df_copy['grau_distancia'] = np.nan
    
    # Para cada thread (target_id único)
    for target_id in df_copy['target_id'].unique():
        # Filtrar dados da thread atual
        thread_df = df_copy[df_copy['target_id'] == target_id].copy()
        
        # Identificar comentários originais (raiz) - MÉTODO MELHORADO
        comentarios_originais = []
        
        # Critério 1: parent_label como "Alvo da Conversa"
        alvo_conversa = thread_df[thread_df['parent_label'] == 'Alvo da Conversa']['id'].tolist()
        comentarios_originais.extend(alvo_conversa)
        
        # Critério 2: parent_name como 'Raiz'
        raiz = thread_df[thread_df['parent_name'] == 'Raiz']['id'].tolist()
        comentarios_originais.extend(raiz)
        
        # Critério 3: parent_id é nulo (primeiro nível)
        sem_parent = thread_df[thread_df['parent_id'].isnull()]['id'].tolist()
        comentarios_originais.extend(sem_parent)
        
        # Critério 4: Nós sem entrada no grafo (raízes topológicas)
        G_temp = nx.DiGraph()
        for _, row in thread_df.iterrows():
            G_temp.add_node(row['id'])
        for _, row in thread_df.iterrows():
            if pd.notnull(row['parent_id']):
                G_temp.add_edge(row['parent_id'], row['id'])
        
        raizes_topologicas = [n for n in G_temp.nodes() if G_temp.in_degree(n) == 0]
        comentarios_originais.extend(raizes_topologicas)
        
        # Critério 5: Comentários com parent_id que não está na thread (comentários originais externos)
        parent_ids_externos = []
        for _, row in thread_df.iterrows():
            if pd.notnull(row['parent_id']) and row['parent_id'] not in thread_df['id'].values:
                parent_ids_externos.append(row['parent_id'])
        
        # Adicionar os comentários externos como originais se não estiverem já incluídos
        for parent_id in parent_ids_externos:
            if parent_id not in comentarios_originais:
                comentarios_originais.append(parent_id)
        
        # Remover duplicatas
        comentarios_originais = list(set(comentarios_originais))
        
        print(f"Thread {target_id}: {len(comentarios_originais)} comentários originais identificados")
        if len(comentarios_originais) > 0:
            print(f"  IDs: {comentarios_originais[:5]}...")  # Mostrar primeiros 5
        
        # Criar grafo para esta thread
        G = nx.DiGraph()
        
        # Adicionar nós
        for _, row in thread_df.iterrows():
            G.add_node(row['id'])
        
        # Adicionar arestas (parent -> filho)
        for _, row in thread_df.iterrows():
            if pd.notnull(row['parent_id']):
                G.add_edge(row['parent_id'], row['id'])
        
        # Calcular distância para cada comentário em relação aos comentários originais
        for _, row in thread_df.iterrows():
            comentario_id = row['id']
            min_distancia = float('inf')
            
            # Calcular distância mínima para qualquer comentário original
            for comentario_original in comentarios_originais:
                try:
                    # Se o comentário original está na thread atual
                    if comentario_original in G.nodes():
                        distancia = nx.shortest_path_length(G, comentario_original, comentario_id)
                        min_distancia = min(min_distancia, distancia)
                    else:
                        # Se o comentário original é externo, calcular distância + 1
                        # (assumindo que é o pai direto de algum comentário na thread)
                        if comentario_original in parent_ids_externos:
                            # Encontrar comentários que têm este parent_id externo
                            filhos_diretos = thread_df[thread_df['parent_id'] == comentario_original]['id'].tolist()
                            for filho in filhos_diretos:
                                if filho in G.nodes():
                                    try:
                                        distancia = nx.shortest_path_length(G, filho, comentario_id) + 1
                                        min_distancia = min(min_distancia, distancia)
                                    except nx.NetworkXNoPath:
                                        continue
                except nx.NetworkXNoPath:
                    # Se não há caminho, continuar
                    continue
            
            # Se encontrou um caminho, atualizar o DataFrame
            if min_distancia != float('inf'):
                # Encontrar o índice no DataFrame original
                idx = df_copy[(df_copy['target_id'] == target_id) & (df_copy['id'] == comentario_id)].index
                if len(idx) > 0:
                    df_copy.loc[idx[0], 'grau_distancia'] = min_distancia
            else:
                # Se não há caminho, pode ser o próprio comentário original
                if comentario_id in comentarios_originais:
                    df_copy.loc[df_copy[(df_copy['target_id'] == target_id) & (df_copy['id'] == comentario_id)].index[0], 'grau_distancia'] = 0
    
    return df_copy

def analisar_distribuicao_graus(df):
    """
    Analisa a distribuição dos graus de distância
    """
    print("=== ANÁLISE DA DISTRIBUIÇÃO DOS GRAUS DE DISTÂNCIA ===")
    
    # Estatísticas básicas
    print(f"Total de comentários: {len(df)}")
    print(f"Comentários com grau calculado: {df['grau_distancia'].notna().sum()}")
    print(f"Comentários sem grau calculado: {df['grau_distancia'].isna().sum()}")
    
    # Distribuição dos graus
    if df['grau_distancia'].notna().sum() > 0:
        print("\nDistribuição dos graus de distância:")
        graus_distribuicao = df['grau_distancia'].value_counts().sort_index()
        for grau, count in graus_distribuicao.items():
            print(f"  Grau {int(grau)}: {count} comentários ({count/len(df)*100:.1f}%)")
        
        print(f"\nGrau máximo: {df['grau_distancia'].max()}")
        print(f"Grau médio: {df['grau_distancia'].mean():.2f}")
        print(f"Grau mediano: {df['grau_distancia'].median():.2f}")
    
    # Análise por thread
    print("\n=== ANÁLISE POR THREAD ===")
    for target_id in df['target_id'].unique():
        thread_df = df[df['target_id'] == target_id]
        thread_com_grau = thread_df[thread_df['grau_distancia'].notna()]
        
        if len(thread_com_grau) > 0:
            print(f"Thread {target_id}: {len(thread_com_grau)}/{len(thread_df)} comentários com grau calculado")
            print(f"  Grau máximo: {thread_com_grau['grau_distancia'].max()}")
            print(f"  Grau médio: {thread_com_grau['grau_distancia'].mean():.2f}")

def main():
    """
    Função principal para executar o cálculo de grau de distância melhorado
    """
    print("Carregando dados...")
    
    try:
        # Tentar carregar o arquivo de teste primeiro
        df = pd.read_csv('Dados/conjuntoDeDados_teste.tsv', sep='\t', encoding='UTF-8')
        print(f"Dados de teste carregados: {len(df)} linhas")
    except Exception as e:
        try:
            # Tentar carregar arquivo de treinamento
            df = pd.read_csv('Dados/conjuntoDeDados_treinamento.tsv', sep='\t', encoding='UTF-8')
            print(f"Dados de treinamento carregados: {len(df)} linhas")
        except Exception as e2:
            try:
                # Tentar carregar arquivo principal
                df = pd.read_csv('Dados/conjuntoDeDados.tsv', sep='\t', encoding='UTF-8')
                print(f"Dados principais carregados: {len(df)} linhas")
            except Exception as e3:
                print(f"Erro ao carregar qualquer arquivo: {e3}")
                return
    
    print("\nColunas disponíveis:")
    print(df.columns.tolist())
    
    print("\nCalculando graus de distância (versão melhorada)...")
    df_com_grau = calcular_grau_distancia_v2(df)
    
    print("\nAnálise dos resultados:")
    analisar_distribuicao_graus(df_com_grau)
    
    # Salvar resultado
    output_file = 'Dados/conjuntoDeDados_com_grau_distancia_v2.tsv'
    df_com_grau.to_csv(output_file, sep='\t', index=False, encoding='UTF-8')
    print(f"\nResultado salvo em: {output_file}")
    
    # Mostrar algumas linhas de exemplo
    print("\nExemplos de comentários com grau de distância:")
    exemplos = df_com_grau[df_com_grau['grau_distancia'].notna()][['id', 'target_id', 'grau_distancia', 'message']].head(10)
    print(exemplos)
    
    return df_com_grau

if __name__ == "__main__":
    df_resultado = main()
