import pandas as pd
import networkx as nx
import numpy as np

class DepthCalculator:
    """
    Classe para calcular o grau de distância de comentários em threads hierárquicas.
    """
    
    def __init__(self, df=None):
        """
        Inicializa o calculador de profundidade.
        
        Args:
            df: DataFrame opcional com os dados para análise
        """
        self.df = df
        self.df_with_depth = None
        
    def set_data(self, df):
        """
        Define os dados para análise.
        
        Args:
            df: DataFrame com as colunas: id, parent_id, target_id, parent_label, parent_name
        """
        self.df = df.copy()
        
    def calcular_grau_distancia(self, df=None):
        """
        Calcula o grau de distância de cada comentário em relação ao comentário original da thread.
        
        Args:
            df: DataFrame opcional. Se não fornecido, usa self.df
        
        Returns:
            DataFrame com nova coluna 'grau_distancia' adicionada
        """
        if df is None:
            if self.df is None:
                raise ValueError("Nenhum DataFrame fornecido. Use set_data() ou forneça df como parâmetro.")
            df = self.df
        
        df_copy = df.copy()
        
        # Adicionar coluna de grau de distância
        df_copy['grau_distancia'] = np.nan
        
        # Para cada thread (target_id único)
        for target_id in df_copy['target_id'].unique():
            # Filtrar dados da thread atual
            thread_df = df_copy[df_copy['target_id'] == target_id].copy()
            
            # Identificar comentários originais (raiz)
            # Comentários originais têm parent_label como "Alvo da Conversa" ou parent_name como 'Raiz'
            comentarios_originais = thread_df[
                (thread_df['parent_label'] == 'Alvo da Conversa') | 
                (thread_df['parent_name'] == 'Raiz')
            ]['id'].tolist()
            
            if not comentarios_originais:
                # Se não encontrar comentários originais pelos critérios acima,
                # procurar por comentários sem parent_id (primeiro nível)
                comentarios_originais = thread_df[thread_df['parent_id'].isnull()]['id'].tolist()
            
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
                        # Calcular caminho mais curto
                        distancia = nx.shortest_path_length(G, comentario_original, comentario_id)
                        min_distancia = min(min_distancia, distancia)
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
        
        # Salvar resultado na instância
        self.df_with_depth = df_copy
        return df_copy

    def analisar_distribuicao_graus(self, df=None):
        """
        Analisa a distribuição dos graus de distância
        
        Args:
            df: DataFrame opcional. Se não fornecido, usa self.df_with_depth
        """
        if df is None:
            if self.df_with_depth is None:
                raise ValueError("Nenhum DataFrame com grau de distância disponível. Execute calcular_grau_distancia() primeiro.")
            df = self.df_with_depth
            
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

    def processar_arquivo(self):
        """
        Processa um arquivo TSV completo: carrega, calcula grau de distância, analisa e salva.
        
        Args:
            file_name: Nome do arquivo (sem extensão)
            input_dir: Diretório de entrada
            output_dir: Diretório de saída
            
        Returns:
            DataFrame com grau de distância calculado
        """
        df = self.df
        print("\nColunas disponíveis:")
        print(df.columns.tolist())
        
        # Definir dados na instância
        self.set_data(df)
        
        print("\nCalculando graus de distância...")
        df_com_grau = self.calcular_grau_distancia()
        
        print("\nAnálise dos resultados:")
        self.analisar_distribuicao_graus()
        
        # Mostrar algumas linhas de exemplo
        print("\nExemplos de comentários com grau de distância:")
        exemplos = df_com_grau[df_com_grau['grau_distancia'].notna()][['id', 'target_id', 'grau_distancia', 'message']].head(10)
        print(exemplos)
        
        return df_com_grau
    
    def salvar_resultado(self, output_file, df=None):
        """
        Salva o DataFrame com grau de distância em arquivo.
        
        Args:
            output_file: Caminho do arquivo de saída
            df: DataFrame opcional. Se não fornecido, usa self.df_with_depth
        """
        if df is None:
            if self.df_with_depth is None:
                raise ValueError("Nenhum DataFrame com grau de distância disponível.")
            df = self.df_with_depth
            
        df.to_csv(output_file, sep='\t', index=False, encoding='UTF-8')
        print(f"Resultado salvo em: {output_file}")
    
    def get_data_with_depth(self):
        """
        Retorna o DataFrame com grau de distância calculado.
        
        Returns:
            DataFrame com coluna 'grau_distancia' ou None se não foi calculado
        """
        return self.df_with_depth
    
    def get_statistics(self, df=None):
        """
        Retorna estatísticas básicas dos graus de distância.
        
        Args:
            df: DataFrame opcional
            
        Returns:
            Dicionário com estatísticas
        """
        if df is None:
            if self.df_with_depth is None:
                raise ValueError("Nenhum DataFrame com grau de distância disponível.")
            df = self.df_with_depth
            
        stats = {
            'total_comentarios': len(df),
            'comentarios_com_grau': df['grau_distancia'].notna().sum(),
            'comentarios_sem_grau': df['grau_distancia'].isna().sum(),
            'grau_maximo': df['grau_distancia'].max() if df['grau_distancia'].notna().sum() > 0 else None,
            'grau_medio': df['grau_distancia'].mean() if df['grau_distancia'].notna().sum() > 0 else None,
            'grau_mediano': df['grau_distancia'].median() if df['grau_distancia'].notna().sum() > 0 else None
        }
        
        return stats


def main():
    """
    Função principal para executar o cálculo de grau de distância (versão legacy)
    """
    print("Usando versão legacy da função main(). Recomenda-se usar a classe DepthCalculator.")
    

    df = pd.read_csv('Dados/conjuntoDeDados_teste.tsv', sep='\t', decimal=',', encoding='UTF-8')
    calculator = DepthCalculator(df)
    df_resultado = calculator.processar_arquivo()
    return df_resultado


if __name__ == "__main__":
    df_resultado = main()
