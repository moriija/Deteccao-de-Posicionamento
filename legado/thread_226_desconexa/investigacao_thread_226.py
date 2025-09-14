import pandas as pd
import networkx as nx
import numpy as np

def investigar_thread_226_detalhado():
    """
    Investigação detalhada da thread 226 para entender por que os graus não estão sendo calculados
    """
    print("=== INVESTIGAÇÃO DETALHADA DA THREAD 226 ===")
    
    # Carregar dados
    df = pd.read_csv('Dados/conjuntoDeDados_teste.tsv', sep='\t', encoding='UTF-8')
    print(f"Dados carregados: {len(df)} linhas")
    
    # Filtrar thread 226
    thread_226 = df[df['target_id'] == 226].copy()
    print(f"\nThread 226: {len(thread_226)} comentários")
    
    # Mostrar estrutura completa
    print("\n=== ESTRUTURA COMPLETA DA THREAD 226 ===")
    for idx, row in thread_226.iterrows():
        print(f"ID: {row['id']}, Parent: {row['parent_id']}, Parent Label: {row['parent_label']}, Parent Name: {row['parent_name']}")
    
    # Criar grafo
    G = nx.DiGraph()
    
    # Adicionar nós
    for _, row in thread_226.iterrows():
        G.add_node(row['id'])
    
    # Adicionar arestas
    for _, row in thread_226.iterrows():
        if pd.notnull(row['parent_id']):
            G.add_edge(row['parent_id'], row['id'])
    
    print(f"\n=== ANÁLISE DO GRAFO ===")
    print(f"Nós: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")
    
    # Encontrar raízes (nós sem entrada)
    raizes = [n for n in G.nodes() if G.in_degree(n) == 0]
    print(f"Raízes (nós sem entrada): {raizes}")
    
    # Encontrar folhas (nós sem saída)
    folhas = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(f"Folhas (nós sem saída): {len(folhas)}")
    
    # Verificar conectividade
    print(f"Grafo é conectado: {nx.is_weakly_connected(G)}")
    print(f"Componentes conectados: {nx.number_weakly_connected_components(G)}")
    
    # Mostrar componentes conectados
    componentes = list(nx.weakly_connected_components(G))
    print(f"\n=== COMPONENTES CONECTADOS ===")
    for i, comp in enumerate(componentes):
        print(f"Componente {i+1}: {len(comp)} nós")
        print(f"  Nós: {list(comp)[:5]}...")  # Primeiros 5 nós
    
    # Testar cálculo de distância manualmente
    print(f"\n=== TESTE MANUAL DE CÁLCULO DE DISTÂNCIA ===")
    
    # Identificar comentários originais
    comentarios_originais = []
    
    # Critério 1: parent_name == 'Raiz'
    raiz = thread_226[thread_226['parent_name'] == 'Raiz']['id'].tolist()
    comentarios_originais.extend(raiz)
    print(f"Comentários com parent_name 'Raiz': {raiz}")
    
    # Critério 2: parent_id externo (não está na thread)
    parent_ids_externos = []
    for _, row in thread_226.iterrows():
        if pd.notnull(row['parent_id']) and row['parent_id'] not in thread_226['id'].values:
            parent_ids_externos.append(row['parent_id'])
    
    parent_ids_externos = list(set(parent_ids_externos))
    comentarios_originais.extend(parent_ids_externos)
    print(f"Parent IDs externos: {parent_ids_externos}")
    
    print(f"Total de comentários originais identificados: {len(comentarios_originais)}")
    print(f"IDs: {comentarios_originais}")
    
    # Testar cálculo para alguns comentários
    print(f"\n=== TESTE DE CÁLCULO PARA ALGUNS COMENTÁRIOS ===")
    
    for i, comentario_id in enumerate(thread_226['id'].head(5)):
        print(f"\nComentário {i+1}: {comentario_id}")
        min_distancia = float('inf')
        
        for comentario_original in comentarios_originais:
            try:
                if comentario_original in G.nodes():
                    # Comentário original está na thread
                    distancia = nx.shortest_path_length(G, comentario_original, comentario_id)
                    print(f"  Distância para {comentario_original}: {distancia}")
                    min_distancia = min(min_distancia, distancia)
                else:
                    # Comentário original é externo
                    print(f"  {comentario_original} é externo")
                    # Encontrar filhos diretos deste parent externo
                    filhos_diretos = thread_226[thread_226['parent_id'] == comentario_original]['id'].tolist()
                    print(f"    Filhos diretos: {filhos_diretos}")
                    
                    for filho in filhos_diretos:
                        if filho in G.nodes():
                            try:
                                distancia = nx.shortest_path_length(G, filho, comentario_id) + 1
                                print(f"    Distância via {filho}: {distancia}")
                                min_distancia = min(min_distancia, distancia)
                            except nx.NetworkXNoPath:
                                print(f"    Sem caminho via {filho}")
            except nx.NetworkXNoPath:
                print(f"  Sem caminho para {comentario_original}")
        
        if min_distancia != float('inf'):
            print(f"  Distância mínima final: {min_distancia}")
        else:
            print(f"  Nenhum caminho encontrado!")

if __name__ == "__main__":
    investigar_thread_226_detalhado()
