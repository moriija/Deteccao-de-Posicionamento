import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def carregar_dados():
    """
    Carrega os dados de treinamento e teste que já incluem os graus de distância
    """
    print("Carregando dados...")
    
    # Carregar dados de treinamento e teste com graus de distância
    df_treino = pd.read_csv('Dados/conjuntoDeDados_treinamento_grau_distancia.tsv', sep='\t', encoding='UTF-8')
    df_teste = pd.read_csv('Dados/conjuntoDeDados_teste_grau_distancia.tsv', sep='\t', encoding='UTF-8')
    
    # Carregar embeddings e features
    dados_treino = joblib.load('input/input_conjuntoDeDados_treinamento.joblib')
    dados_teste = joblib.load('input/input_conjuntoDeDados_teste.joblib')
    
    print(f"Dados de treinamento: {len(dados_treino['features'])} amostras")
    print(f"Dados de teste: {len(dados_teste['features'])} amostras")
    print(f"Graus únicos treino: {sorted(df_treino['grau_distancia'].unique())}")
    print(f"Graus únicos teste: {sorted(df_teste['grau_distancia'].unique())}")
    
    return dados_treino, dados_teste, df_treino, df_teste

def treinar_modelo_base(dados_treino):
    """
    Treina o modelo Random Forest base usando as features originais
    """
    print("\nTreinando modelo base...")
    
    # Features originais: embedding do comentário + embedding do target + parent_label real
    features_base = np.concatenate((dados_treino['features'], dados_treino['parent_label']), axis=1)
    
    modelo_base = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_base.fit(features_base, dados_treino['target'])
    
    print("Modelo base treinado com sucesso!")
    return modelo_base

def classificar_hierarquicamente(dados_treino, dados_teste, df_treino, df_teste, modelo_base):
    """
    Realiza classificação hierárquica usando as previsões dos comentários pais
    """
    print("\nIniciando classificação hierárquica...")
    
    # Criar cópias dos dados para não modificar os originais
    features_treino = dados_treino['features'].copy()
    features_teste = dados_teste['features'].copy()
    
    # Obter graus de distância diretamente dos DataFrames
    graus_treino = df_treino['grau_distancia'].values
    graus_teste = df_teste['grau_distancia'].values
    
    print(f"Graus de distância obtidos:")
    print(f"  Treinamento: {len(graus_treino)} amostras, graus: {sorted(set(graus_treino))}")
    print(f"  Teste: {len(graus_teste)} amostras, graus: {sorted(set(graus_teste))}")
    
    # Ordenar dados por grau de distância (do menor para o maior)
    indices_treino_ordenados = np.argsort(graus_treino)
    indices_teste_ordenados = np.argsort(graus_teste)
    
    # Reorganizar dados
    features_treino_ordenadas = features_treino[indices_treino_ordenados]
    features_teste_ordenadas = features_teste[indices_teste_ordenados]
    targets_treino_ordenados = dados_treino['target'][indices_treino_ordenados]
    targets_teste_ordenados = dados_teste['target'][indices_teste_ordenados]
    graus_treino_ordenados = graus_treino[indices_treino_ordenados]
    graus_teste_ordenados = graus_teste[indices_teste_ordenados]
    
    # Inicializar array de previsões hierárquicas
    predicoes_hierarquicas_treino = np.full(len(features_treino), -999)
    predicoes_hierarquicas_teste = np.full(len(features_teste), -999)
    
    # Dicionário para armazenar previsões por grau
    predicoes_por_grau = {}
    
    # Classificar por grau de distância
    graus_unicos = sorted(set(np.concatenate([graus_treino_ordenados, graus_teste_ordenados])))
    
    for grau in graus_unicos:
        print(f"\nProcessando grau {grau}...")
        
        # Índices para este grau
        idx_treino_grau = np.where(graus_treino_ordenados == grau)[0]
        idx_teste_grau = np.where(graus_teste_ordenados == grau)[0]
        
        if len(idx_treino_grau) == 0 and len(idx_teste_grau) == 0:
            continue
        
        # Features para este grau
        features_grau_treino = features_treino_ordenadas[idx_treino_grau]
        features_grau_teste = features_teste_ordenadas[idx_teste_grau]
        
        # Targets para este grau
        targets_grau_treino = targets_treino_ordenados[idx_treino_grau]
        
        # Se é grau 0, usar modelo base
        if grau == 0:
            modelo_grau = modelo_base
        else:
            # Para graus maiores, criar features com previsões dos pais
            features_com_predicoes_pais_treino = []
            features_com_predicoes_pais_teste = []
            
            # Para treinamento
            for i, idx in enumerate(idx_treino_grau):
                # Encontrar comentários pais (grau anterior)
                grau_pai = grau - 1
                idx_pai = np.where(graus_treino_ordenados == grau_pai)[0]
                
                if len(idx_pai) > 0:
                    # Usar a média das previsões dos pais (simplificação)
                    predicoes_pais = predicoes_hierarquicas_treino[idx_pai]
                    predicoes_pais = predicoes_pais[predicoes_pais != -999]
                    
                    if len(predicoes_pais) > 0:
                        # Média das previsões dos pais
                        pred_media_pais = np.mean(predicoes_pais)
                        # Concatenar com features originais
                        feature_com_pred = np.concatenate([features_grau_treino[i], [pred_media_pais]])
                        features_com_predicoes_pais_treino.append(feature_com_pred)
                    else:
                        # Se não há previsões dos pais, usar valor padrão
                        feature_com_pred = np.concatenate([features_grau_treino[i], [0]])
                        features_com_predicoes_pais_treino.append(feature_com_pred)
                else:
                    # Se não há pais, usar valor padrão
                    feature_com_pred = np.concatenate([features_grau_treino[i], [0]])
                    features_com_predicoes_pais_treino.append(feature_com_pred)
            
            # Para teste
            for i, idx in enumerate(idx_teste_grau):
                # Encontrar comentários pais (grau anterior)
                grau_pai = grau - 1
                idx_pai = np.where(graus_teste_ordenados == grau_pai)[0]
                
                if len(idx_pai) > 0:
                    # Usar a média das previsões dos pais
                    predicoes_pais = predicoes_hierarquicas_teste[idx_pai]
                    predicoes_pais = predicoes_pais[predicoes_pais != -999]
                    
                    if len(predicoes_pais) > 0:
                        pred_media_pais = np.mean(predicoes_pais)
                        feature_com_pred = np.concatenate([features_grau_teste[i], [pred_media_pais]])
                        features_com_predicoes_pais_teste.append(feature_com_pred)
                    else:
                        feature_com_pred = np.concatenate([features_grau_teste[i], [0]])
                        features_com_predicoes_pais_teste.append(feature_com_pred)
                else:
                    feature_com_pred = np.concatenate([features_grau_teste[i], [0]])
                    features_com_predicoes_pais_teste.append(feature_com_pred)
            
            # Treinar modelo para este grau
            if len(features_com_predicoes_pais_treino) > 0:
                features_com_predicoes_pais_treino = np.array(features_com_predicoes_pais_treino)
                modelo_grau = RandomForestClassifier(n_estimators=100, random_state=42)
                modelo_grau.fit(features_com_predicoes_pais_treino, targets_grau_treino)
            else:
                modelo_grau = modelo_base
        
        # Fazer previsões
        if grau == 0:
            # Para grau 0, usar features originais + parent_label (como no treinamento)
            features_grau_treino_com_parent = np.concatenate((features_grau_treino, 
                                                           dados_treino['parent_label'][idx_treino_grau]), axis=1)
            features_grau_teste_com_parent = np.concatenate((features_grau_teste, 
                                                          dados_teste['parent_label'][idx_teste_grau]), axis=1)
            
            pred_treino = modelo_grau.predict(features_grau_treino_com_parent)
            pred_teste = modelo_grau.predict(features_grau_teste_com_parent)
        else:
            # Para graus maiores, usar features com previsões dos pais
            if len(features_com_predicoes_pais_treino) > 0:
                pred_treino = modelo_grau.predict(features_com_predicoes_pais_treino)
                # Verificar se há features de teste válidas
                if len(features_com_predicoes_pais_teste) > 0:
                    pred_teste = modelo_grau.predict(features_com_predicoes_pais_teste)
                else:
                    # Se não há features de teste, criar array vazio
                    pred_teste = np.array([])
            else:
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

def avaliar_desempenho_por_grau(predicoes_por_grau):
    """
    Avalia o desempenho do modelo por grau de distância
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

def comparar_modelos(dados_teste, predicoes_hierarquicas_teste, modelo_base):
    """
    Compara o desempenho do modelo hierárquico com o modelo base
    """
    print("\n" + "="*60)
    print("COMPARAÇÃO: MODELO BASE vs MODELO HIERÁRQUICO")
    print("="*60)
    
    # Previsões do modelo base
    features_base_teste = np.concatenate((dados_teste['features'], dados_teste['parent_label']), axis=1)
    predicoes_base = modelo_base.predict(features_base_teste)
    
    # Avaliar modelo base
    acuracia_base = accuracy_score(dados_teste['target'], predicoes_base)
    relatorio_base = classification_report(dados_teste['target'], predicoes_base, output_dict=True)
    
    # Avaliar modelo hierárquico
    acuracia_hierarquico = accuracy_score(dados_teste['target'], predicoes_hierarquicas_teste)
    relatorio_hierarquico = classification_report(dados_teste['target'], predicoes_hierarquicas_teste, output_dict=True)
    
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

def main():
    """
    Função principal
    """
    print("CLASSIFICAÇÃO HIERÁRQUICA COM RANDOM FOREST")
    print("="*50)
    
    # Carregar dados
    dados_treino, dados_teste, df_treino, df_teste = carregar_dados()
    
    # Treinar modelo base
    modelo_base = treinar_modelo_base(dados_treino)
    
    # Realizar classificação hierárquica
    predicoes_por_grau, pred_treino_hier, pred_teste_hier = classificar_hierarquicamente(
        dados_treino, dados_teste, df_treino, df_teste, modelo_base
    )
    
    # Avaliar desempenho por grau
    resultados_por_grau = avaliar_desempenho_por_grau(predicoes_por_grau)
    
    # Comparar modelos
    comparacao = comparar_modelos(dados_teste, pred_teste_hier, modelo_base)
    
    # Salvar resultados
    resultados = {
        'predicoes_por_grau': predicoes_por_grau,
        'resultados_por_grau': resultados_por_grau,
        'comparacao_modelos': comparacao,
        'predicoes_hierarquicas_teste': pred_teste_hier
    }
    
    joblib.dump(resultados, 'resultados_classificacao_hierarquica.joblib')
    print(f"\nResultados salvos em: resultados_classificacao_hierarquica.joblib")
    
    return resultados

if __name__ == "__main__":
    resultados = main()
