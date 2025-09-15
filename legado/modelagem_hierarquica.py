# -*- coding: utf-8 -*-
"""
Classe para modelagem hierárquica de detecção de posicionamento.
Implementa classificação hierárquica usando graus de distância e Random Forest.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class ModelagemHierarquica:
    """
    Classe para modelagem hierárquica de detecção de posicionamento.
    
    Funcionalidades:
    - Treinamento de modelo base
    - Classificação hierárquica por grau de distância
    - Avaliação de desempenho por grau
    - Comparação entre modelos
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Inicializa a classe de modelagem hierárquica.
        
        Args:
            n_estimators: Número de estimadores para Random Forest
            random_state: Seed para reprodutibilidade
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.modelo_base = None
        
    def treinar_modelo_base(self, features, targets, parent_labels):
        """
        Treina o modelo Random Forest base.
        
        Args:
            features: Features combinadas (embeddings)
            targets: Labels de destino
            parent_labels: Labels dos comentários pais
            
        Returns:
            Modelo treinado
        """
        print("Treinando modelo base...")
        
        # Features: embeddings + parent_label
        features_completas = np.concatenate((features, parent_labels), axis=1)
        
        self.modelo_base = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        self.modelo_base.fit(features_completas, targets)
        
        print("Modelo base treinado com sucesso!")
        return self.modelo_base
    
    def classificar_hierarquicamente(self, df_treino, df_teste, features_treino, features_teste, 
                                   targets_treino, targets_teste, parent_labels_treino, parent_labels_teste):
        """
        Realiza classificação hierárquica usando graus de distância.
        
        Args:
            df_treino: DataFrame de treinamento com graus de distância
            df_teste: DataFrame de teste com graus de distância
            features_treino: Features de treinamento
            features_teste: Features de teste
            targets_treino: Targets de treinamento
            targets_teste: Targets de teste
            parent_labels_treino: Parent labels de treinamento
            parent_labels_teste: Parent labels de teste
            
        Returns:
            dict: Resultados da classificação hierárquica
        """
        print("Iniciando classificação hierárquica...")
        
        # Obter graus de distância
        graus_treino = df_treino['grau_distancia'].values
        graus_teste = df_teste['grau_distancia'].values
        
        print(f"Graus de distância obtidos:")
        print(f"  Treinamento: {len(graus_treino)} amostras, graus: {sorted(set(graus_treino))}")
        print(f"  Teste: {len(graus_teste)} amostras, graus: {sorted(set(graus_teste))}")
        
        # Ordenar por grau de distância
        indices_treino_ordenados = np.argsort(graus_treino)
        indices_teste_ordenados = np.argsort(graus_teste)
        
        # Reorganizar dados
        features_treino_ordenadas = features_treino[indices_treino_ordenados]
        features_teste_ordenadas = features_teste[indices_teste_ordenados]
        targets_treino_ordenados = targets_treino[indices_treino_ordenados]
        targets_teste_ordenados = targets_teste[indices_teste_ordenados]
        graus_treino_ordenados = graus_treino[indices_treino_ordenados]
        graus_teste_ordenados = graus_teste[indices_teste_ordenados]
        parent_labels_treino_ordenados = parent_labels_treino[indices_treino_ordenados]
        parent_labels_teste_ordenados = parent_labels_teste[indices_teste_ordenados]
        
        # Inicializar arrays de previsões
        predicoes_hierarquicas_treino = np.full(len(features_treino), -999)
        predicoes_hierarquicas_teste = np.full(len(features_teste), -999)
        
        predicoes_por_grau = {}
        
        # Classificar por grau de distância
        graus_unicos = sorted(set(np.concatenate([graus_treino_ordenados, graus_teste_ordenados])))
        
        for grau in graus_unicos:
            print(f"Processando grau {grau}...")
            
            # Índices para este grau
            idx_treino_grau = np.where(graus_treino_ordenados == grau)[0]
            idx_teste_grau = np.where(graus_teste_ordenados == grau)[0]
            
            if len(idx_treino_grau) == 0 and len(idx_teste_grau) == 0:
                continue
            
            # Features para este grau
            features_grau_treino = features_treino_ordenadas[idx_treino_grau]
            features_grau_teste = features_teste_ordenadas[idx_teste_grau]
            targets_grau_treino = targets_treino_ordenados[idx_treino_grau]
            parent_labels_grau_treino = parent_labels_treino_ordenados[idx_treino_grau]
            parent_labels_grau_teste = parent_labels_teste_ordenados[idx_teste_grau]
            
            # Se é grau 0, usar modelo base
            if grau == 0:
                modelo_grau = self.modelo_base
                features_grau_treino_com_parent = np.concatenate((features_grau_treino, parent_labels_grau_treino), axis=1)
                features_grau_teste_com_parent = np.concatenate((features_grau_teste, parent_labels_grau_teste), axis=1)
                
                pred_treino = modelo_grau.predict(features_grau_treino_com_parent)
                pred_teste = modelo_grau.predict(features_grau_teste_com_parent)
            else:
                # Para graus maiores, criar features com previsões dos pais
                features_com_predicoes_pais_treino = []
                features_com_predicoes_pais_teste = []
                
                # Para treinamento
                for i, idx in enumerate(idx_treino_grau):
                    grau_pai = grau - 1
                    idx_pai = np.where(graus_treino_ordenados == grau_pai)[0]
                    
                    if len(idx_pai) > 0:
                        predicoes_pais = predicoes_hierarquicas_treino[idx_pai]
                        predicoes_pais = predicoes_pais[predicoes_pais != -999]
                        
                        if len(predicoes_pais) > 0:
                            pred_media_pais = np.mean(predicoes_pais)
                            feature_com_pred = np.concatenate([features_grau_treino[i], [pred_media_pais]])
                        else:
                            feature_com_pred = np.concatenate([features_grau_treino[i], [0]])
                    else:
                        feature_com_pred = np.concatenate([features_grau_treino[i], [0]])
                    
                    features_com_predicoes_pais_treino.append(feature_com_pred)
                
                # Para teste
                for i, idx in enumerate(idx_teste_grau):
                    grau_pai = grau - 1
                    idx_pai = np.where(graus_teste_ordenados == grau_pai)[0]
                    
                    if len(idx_pai) > 0:
                        predicoes_pais = predicoes_hierarquicas_teste[idx_pai]
                        predicoes_pais = predicoes_pais[predicoes_pais != -999]
                        
                        if len(predicoes_pais) > 0:
                            pred_media_pais = np.mean(predicoes_pais)
                            feature_com_pred = np.concatenate([features_grau_teste[i], [pred_media_pais]])
                        else:
                            feature_com_pred = np.concatenate([features_grau_teste[i], [0]])
                    else:
                        feature_com_pred = np.concatenate([features_grau_teste[i], [0]])
                    
                    features_com_predicoes_pais_teste.append(feature_com_pred)
                
                # Treinar modelo para este grau
                if len(features_com_predicoes_pais_treino) > 0:
                    features_com_predicoes_pais_treino = np.array(features_com_predicoes_pais_treino)
                    modelo_grau = RandomForestClassifier(
                        n_estimators=self.n_estimators, 
                        random_state=self.random_state
                    )
                    modelo_grau.fit(features_com_predicoes_pais_treino, targets_grau_treino)
                    
                    if len(features_com_predicoes_pais_teste) > 0:
                        features_com_predicoes_pais_teste = np.array(features_com_predicoes_pais_teste)
                        pred_treino = modelo_grau.predict(features_com_predicoes_pais_treino)
                        pred_teste = modelo_grau.predict(features_com_predicoes_pais_teste)
                    else:
                        pred_treino = modelo_grau.predict(features_com_predicoes_pais_treino)
                        pred_teste = np.array([])
                else:
                    modelo_grau = self.modelo_base
                    pred_treino = modelo_grau.predict(features_grau_treino)
                    pred_teste = modelo_grau.predict(features_grau_teste)
            
            # Armazenar previsões
            predicoes_hierarquicas_treino[idx_treino_grau] = pred_treino
            predicoes_hierarquicas_teste[idx_teste_grau] = pred_teste
            
            # Armazenar por grau para análise
            predicoes_por_grau[grau] = {
                'treino': pred_treino,
                'teste': pred_teste,
                'targets_treino': targets_grau_treino,
                'targets_teste': targets_teste_ordenados[idx_teste_grau] if len(idx_teste_grau) > 0 else np.array([])
            }
            
            print(f"  Grau {grau}: {len(idx_treino_grau)} amostras de treino, {len(idx_teste_grau)} de teste")
        
        return predicoes_por_grau, predicoes_hierarquicas_treino, predicoes_hierarquicas_teste
    
    def avaliar_desempenho_por_grau(self, predicoes_por_grau):
        """
        Avalia o desempenho do modelo por grau de distância.
        
        Args:
            predicoes_por_grau: Dicionário com previsões por grau
            
        Returns:
            dict: Resultados de avaliação por grau
        """
        print("\n" + "="*60)
        print("AVALIAÇÃO DO DESEMPENHO POR GRAU DE DISTÂNCIA")
        print("="*60)
        
        resultados_por_grau = {}
        
        for grau in sorted(predicoes_por_grau.keys()):
            dados_grau = predicoes_por_grau[grau]
            
            if len(dados_grau['targets_teste']) == 0:
                continue
                
            # Calcular acurácia
            acuracia = accuracy_score(dados_grau['targets_teste'], dados_grau['teste'])
            
            # Relatório de classificação
            relatorio = classification_report(dados_grau['targets_teste'], dados_grau['teste'], 
                                           output_dict=True, zero_division=0)
            
            resultados_por_grau[grau] = {
                'acuracia': acuracia,
                'precision_macro': relatorio['macro avg']['precision'],
                'recall_macro': relatorio['macro avg']['recall'],
                'f1_macro': relatorio['macro avg']['f1-score'],
                'n_amostras': len(dados_grau['targets_teste'])
            }
            
            print(f"\nGrau {grau}:")
            print(f"  Amostras: {len(dados_grau['targets_teste'])}")
            print(f"  Acurácia: {acuracia:.3f}")
            print(f"  Precision (macro): {relatorio['macro avg']['precision']:.3f}")
            print(f"  Recall (macro): {relatorio['macro avg']['recall']:.3f}")
            print(f"  F1-Score (macro): {relatorio['macro avg']['f1-score']:.3f}")
            
            # Relatório detalhado
            print(f"  Relatório detalhado:")
            print(classification_report(dados_grau['targets_teste'], dados_grau['teste'], zero_division=0))
        
        return resultados_por_grau
    
    def comparar_modelos(self, targets_teste, predicoes_hierarquicas_teste, features_teste, parent_labels_teste):
        """
        Compara o desempenho do modelo hierárquico com o modelo base.
        
        Args:
            targets_teste: Targets reais de teste
            predicoes_hierarquicas_teste: Previsões do modelo hierárquico
            features_teste: Features de teste
            parent_labels_teste: Parent labels de teste
            
        Returns:
            dict: Comparação entre modelos
        """
        print("\n" + "="*60)
        print("COMPARAÇÃO: MODELO BASE vs MODELO HIERÁRQUICO")
        print("="*60)
        
        # Previsões do modelo base
        features_base_teste = np.concatenate((features_teste, parent_labels_teste), axis=1)
        predicoes_base = self.modelo_base.predict(features_base_teste)
        
        # Avaliar modelo base
        acuracia_base = accuracy_score(targets_teste, predicoes_base)
        relatorio_base = classification_report(targets_teste, predicoes_base, output_dict=True)
        
        # Avaliar modelo hierárquico
        acuracia_hierarquico = accuracy_score(targets_teste, predicoes_hierarquicas_teste)
        relatorio_hierarquico = classification_report(targets_teste, predicoes_hierarquicas_teste, output_dict=True)
        
        print(f"Modelo Base:")
        print(f"  Acurácia: {acuracia_base:.3f}")
        print(f"  F1-Score (macro): {relatorio_base['macro avg']['f1-score']:.3f}")
        
        print(f"\nModelo Hierárquico:")
        print(f"  Acurácia: {acuracia_hierarquico:.3f}")
        print(f"  F1-Score (macro): {relatorio_hierarquico['macro avg']['f1-score']:.3f}")
        
        print(f"\nDiferença:")
        print(f"  Acurácia: {acuracia_hierarquico - acuracia_base:+.3f}")
        print(f"  F1-Score: {relatorio_hierarquico['macro avg']['f1-score'] - relatorio_base['macro avg']['f1-score']:+.3f}")
        
        return {
            'base': {'acuracia': acuracia_base, 'f1_macro': relatorio_base['macro avg']['f1-score']},
            'hierarquico': {'acuracia': acuracia_hierarquico, 'f1_macro': relatorio_hierarquico['macro avg']['f1-score']}
        }
    
    def executar_modelagem_completa(self, df_treino, df_teste, dados_treino, dados_teste, salvar_resultados=True):
        """
        Executa modelagem hierárquica completa.
        
        Args:
            df_treino: DataFrame de treinamento com graus de distância
            df_teste: DataFrame de teste com graus de distância
            dados_treino: Dados de treinamento (features, target, parent_label)
            dados_teste: Dados de teste (features, target, parent_label)
            salvar_resultados: Se deve salvar resultados em arquivo
            
        Returns:
            dict: Resultados completos da modelagem
        """
        print("EXECUTANDO MODELAGEM HIERÁRQUICA COMPLETA")
        print("="*50)
        
        # Treinar modelo base
        self.treinar_modelo_base(
            dados_treino['features'], 
            dados_treino['target'], 
            dados_treino['parent_label']
        )
        
        # Classificação hierárquica
        predicoes_por_grau, pred_treino_hier, pred_teste_hier = self.classificar_hierarquicamente(
            df_treino, df_teste,
            dados_treino['features'], dados_teste['features'],
            dados_treino['target'], dados_teste['target'],
            dados_treino['parent_label'], dados_teste['parent_label']
        )
        
        # Avaliar desempenho
        resultados_por_grau = self.avaliar_desempenho_por_grau(predicoes_por_grau)
        
        # Comparar modelos
        comparacao = self.comparar_modelos(
            dados_teste['target'], pred_teste_hier,
            dados_teste['features'], dados_teste['parent_label']
        )
        
        # Compilar resultados
        resultados = {
            'predicoes_por_grau': predicoes_por_grau,
            'resultados_por_grau': resultados_por_grau,
            'comparacao_modelos': comparacao,
            'predicoes_hierarquicas_treino': pred_treino_hier,
            'predicoes_hierarquicas_teste': pred_teste_hier,
            'modelo_base': self.modelo_base
        }
        
        # Salvar resultados
        if salvar_resultados:
            joblib.dump(resultados, 'resultados_modelagem_hierarquica.joblib')
            print(f"\nResultados salvos em: resultados_modelagem_hierarquica.joblib")
        
        return resultados
    
    def carregar_dados_para_modelagem(self, nome_arquivo_treino, nome_arquivo_teste):
        """
        Carrega dados processados para modelagem.
        
        Args:
            nome_arquivo_treino: Nome do arquivo de treinamento
            nome_arquivo_teste: Nome do arquivo de teste
            
        Returns:
            tuple: (df_treino, df_teste, dados_treino, dados_teste)
        """
        print("Carregando dados para modelagem...")
        
        # Carregar DataFrames com graus de distância
        df_treino = pd.read_csv(f'Dados/{nome_arquivo_treino}_grau_distancia.tsv', sep='\t', encoding='UTF-8')
        df_teste = pd.read_csv(f'Dados/{nome_arquivo_teste}_grau_distancia.tsv', sep='\t', encoding='UTF-8')
        
        # Carregar dados de input
        dados_treino = joblib.load(f'input/input_{nome_arquivo_treino}.joblib')
        dados_teste = joblib.load(f'input/input_{nome_arquivo_teste}.joblib')
        
        print(f"Dados de treinamento: {len(dados_treino['features'])} amostras")
        print(f"Dados de teste: {len(dados_teste['features'])} amostras")
        print(f"Graus únicos treino: {sorted(df_treino['grau_distancia'].unique())}")
        print(f"Graus únicos teste: {sorted(df_teste['grau_distancia'].unique())}")
        
        return df_treino, df_teste, dados_treino, dados_teste


def main():
    """
    Função principal para demonstração de uso da classe.
    """
    print("MODELAGEM HIERÁRQUICA")
    print("="*30)
    
    # Criar instância da modelagem
    modelagem = ModelagemHierarquica()
    
    # Carregar dados
    df_treino, df_teste, dados_treino, dados_teste = modelagem.carregar_dados_para_modelagem(
        'conjuntoDeDados_treinamento', 
        'conjuntoDeDados_teste'
    )
    
    # Executar modelagem completa
    resultados = modelagem.executar_modelagem_completa(
        df_treino, df_teste, dados_treino, dados_teste
    )
    
    print("\nModelagem concluída!")
    print(f"Graus processados: {len(resultados['resultados_por_grau'])}")
    print(f"Comparação de modelos: {resultados['comparacao_modelos']}")


if __name__ == "__main__":
    main()

