import pandas as pd
import networkx as nx
import numpy as np

def diagnosticar_thread_226():
    """
    Diagnostica especificamente a thread 226 para entender por que os graus não estão sendo calculados
    """
    print("=== DIAGNÓSTICO DA THREAD 226 ===")
    
    # Carregar dados
    try:
        df = pd.read_csv('Dados/conjuntoDeDados_teste.tsv', sep='\t', encoding='UTF-8')
        print(f"Dados carregados: {len(df)} linhas")
    except:
        try:
            df = pd.read_csv('Dados/conjuntoDeDados_treinamento.tsv', sep='\t', encoding='UTF-8')
            print(f"Dados de treinamento carregados: {len(df)} linhas")
        except:
            df = pd.read_csv('Dados/conjuntoDeDados.tsv', sep='\t', encoding='UTF-8')
            print(f"Dados principais carregados: {len(df)} linhas")
    
    # Filtrar thread 226
    thread_226 = df[df['target_id'] == 226].copy()
    print(f"\nThread 226: {len(thread_226)} comentários")
    
    if len(thread_226) == 0:
        print("ERRO: Thread 226 não encontrada!")
        return
    
    # Mostrar todas as colunas da thread 226
    print("\nColunas disponíveis:")
    print(thread_226.columns.tolist())
    
    # Verificar valores únicos em colunas importantes
    print("\n=== ANÁLISE DAS COLUNAS ===")
    print(f"IDs únicos: {thread_226['id'].nunique()}")
    print(f"Parent IDs únicos: {thread_226['parent_id'].nunique()}")
    print(f"Parent labels únicos: {thread_226['parent_label'].unique()}")
    print(f"Parent names únicos: {thread_226['parent_name'].unique()}")
    
    # Mostrar primeiras linhas
    print("\n=== PRIMEIRAS LINHAS DA THREAD 226 ===")
    print(thread_226[['id', 'parent_id', 'parent_label', 'parent_name', 'message']].head(10))
    
    # Verificar se há comentários originais
    print("\n=== IDENTIFICAÇÃO DE COMENTÁRIOS ORIGINAIS ===")
    
    # Critério 1: parent_label == "Alvo da Conversa"
    alvo_conversa = thread_226[thread_226['parent_label'] == 'Alvo da Conversa']
    print(f"Comentários com parent_label 'Alvo da Conversa': {len(alvo_conversa)}")
    if len(alvo_conversa) > 0:
        print("IDs:", alvo_conversa['id'].tolist())
    
    # Critério 2: parent_name == 'Raiz'
    raiz = thread_226[thread_226['parent_name'] == 'Raiz']
    print(f"Comentários com parent_name 'Raiz': {len(raiz)}")
    if len(raiz) > 0:
        print("IDs:", raiz['id'].tolist())
    
    # Critério 3: parent_id é nulo
    sem_parent = thread_226[thread_226['parent_id'].isnull()]
    print(f"Comentários sem parent_id: {len(sem_parent)}")
    if len(sem_parent) > 0:
        print("IDs:", sem_parent['id'].tolist())
    
    # Criar grafo para esta thread
    print("\n=== CRIAÇÃO DO GRAFO ===")
    G = nx.DiGraph()
    
    # Adicionar nós
    for idx, row in thread_226.iterrows():
        G.add_node(row['id'])
    
    # Adicionar arestas
    edges_added = 0
    for idx, row in thread_226.iterrows():
        if pd.notnull(row['parent_id']):
            G.add_edge(row['parent_id'], row['id'])
            edges_added += 1
    
    print(f"Nós adicionados: {G.number_of_nodes()}")
    print(f"Arestas adicionadas: {edges_added}")
    
    # Verificar conectividade
    print(f"Grafo é conectado: {nx.is_weakly_connected(G)}")
    print(f"Componentes conectados: {nx.number_weakly_connected_components(G)}")
    
    # Tentar identificar comentários originais pelo grafo
    print("\n=== ANÁLISE DO GRAFO ===")
    
    # Encontrar nós sem entrada (raízes)
    raizes_grafo = [n for n in G.nodes() if G.in_degree(n) == 0]
    print(f"Nós sem entrada (raízes): {len(raizes_grafo)}")
    if raizes_grafo:
        print("IDs das raízes:", raizes_grafo)
    
    # Encontrar nós sem saída (folhas)
    folhas = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(f"Nós sem saída (folhas): {len(folhas)}")
    
    # Mostrar estrutura completa
    print("\n=== ESTRUTURA COMPLETA DA THREAD 226 ===")
    for idx, row in thread_226.iterrows():
        print(f"ID: {row['id']}, Parent: {row['parent_id']}, Parent Label: {row['parent_label']}, Parent Name: {row['parent_name']}")
        print(f"  Mensagem: {row['message'][:100]}...")
        print()

if __name__ == "__main__":
    diagnosticar_thread_226()
