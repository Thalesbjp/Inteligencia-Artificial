import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, silhouette_score
from scipy.optimize import linear_sum_assignment

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")


def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega a base de dados Iris a partir do arquivo CSV local.
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"O arquivo '{caminho_arquivo}' não foi encontrado na pasta.")
    
    df = pd.read_csv(caminho_arquivo)
    print(f"[INFO] Base de dados '{caminho_arquivo}' carregada com sucesso. Total de instâncias: {len(df)}")
    return df


def pre_processar_dados(X: np.ndarray, opcao: int) -> np.ndarray:
    """
    Realiza o pré-processamento dos atributos com base na escolha do usuário:
    - Opção 0 (Sem pré-processamento): Mantém os dados na escala original.
    - Opção 1 (Normalização / MinMaxScaler): Coloca os valores no intervalo [0, 1].
    - Opção 2 (Padronização / StandardScaler): Centraliza a média em 0 e desvio padrão em 1.
    """
    if opcao == 0:
        print("[INFO] Nenhum pré-processamento aplicado (utilizando dados brutos).")
        return X
    elif opcao == 1:
        print("[INFO] Aplicando Normalização (MinMaxScaler - intervalo [0, 1])...")
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    elif opcao == 2:
        print("[INFO] Aplicando Padronização (StandardScaler - média 0, variância 1)...")
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    else:
        print("[AVISO] Opção inválida. Retornando os dados sem pré-processamento.")
        return X


def alinhar_clusters_com_rotulos(y_verdadeiro: np.ndarray, y_predito: np.ndarray) -> np.ndarray:
    """
    Como o K-Means é não supervisionado, ele retorna rótulos arbitrários (ex: 0, 1, 2) 
    que podem não corresponder diretamente às classes reais. Esta função usa o algoritmo 
    de Hungarian para mapear os clusters aos rótulos reais de forma ótima para as métricas.
    """
    matriz_contingencia = confusion_matrix(y_verdadeiro, y_predito)
    linha_ind, col_ind = linear_sum_assignment(-matriz_contingencia)
    mapeamento = {col: linha for linha, col in zip(linha_ind, col_ind)}
    y_mapeado = np.array([mapeamento[cluster] for cluster in y_predito])
    return y_mapeado


def avaliar_agrupamentos_e_classificador(X: np.ndarray, y_verdadeiro_str: np.ndarray, y_predito: np.ndarray, k: int, kmeans_model: KMeans):
    """
    Calcula as métricas de agrupamento (Inércia/WCSS, Silhouette Score), 
    métricas de classificação (Acurácia, Precisão, Revocação) e a Matriz de Confusão.
    """
    # 1. Métricas de Agrupamento
    wcss = kmeans_model.inertia_
    
    # Silhouette Score só pode ser calculado se K >= 2
    if k > 1:
        sil_score = silhouette_score(X, y_predito)
    else:
        sil_score = float('nan')
        
    print("\n" + "="*50)
    print("        AVALIAÇÃO DE AGRUPAMENTO (K-MEANS)        ")
    print("="*50)
    print(f"Número de Clusters (K) : {k}")
    print(f"Inércia (WCSS)         : {wcss:.4f}")
    print(f"Silhouette Score       : {sil_score:.4f} (Próximo de +1 indica grupos bem separados)")
    
    # 2. Conversão e Alinhamento para Métricas de Classificação
    classes_unicas = np.unique(y_verdadeiro_str)
    mapeamento_classes = {nome: i for i, nome in enumerate(classes_unicas)}
    y_verdadeiro = np.array([mapeamento_classes[nome] for nome in y_verdadeiro_str])
    
    y_predito_alinhado = alinhar_clusters_com_rotulos(y_verdadeiro, y_predito)
    
    acuracia = accuracy_score(y_verdadeiro, y_predito_alinhado)
    precisao = precision_score(y_verdadeiro, y_predito_alinhado, average='macro', zero_division=0)
    revocacao = recall_score(y_verdadeiro, y_predito_alinhado, average='macro', zero_division=0)
    
    print("-" * 50)
    print("        AVALIAÇÃO DE DESEMPENHO (EXTERNA)       ")
    print(f"Acurácia Global  : {acuracia * 100:.2f}%")
    print(f"Precisão (Macro) : {precisao * 100:.2f}%")
    print(f"Revocação (Macro): {revocacao * 100:.2f}%")
    print("-" * 50)
    
    # 3. Matriz de Confusão
    cm = confusion_matrix(y_verdadeiro, y_predito_alinhado)
    print("Matriz de Confusão (Linhas: Reais | Colunas: Clusters Alinhados):")
    print(cm)
    print("="*50)
    
    # Plotagem da Matriz de Confusão
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_unicas, 
                yticklabels=classes_unicas)
    plt.title(f'Matriz de Confusão (K={k})')
    plt.xlabel('Clusters Preditos (Alinhados)')
    plt.ylabel('Classes Reais')
    plt.tight_layout()
    plt.savefig('matriz_confusao_iris.png')
    print("[INFO] Gráfico da matriz de confusão salvo como 'matriz_confusao_iris.png'.")
    plt.show()


def metodo_cotovelo_e_silueta(X: np.ndarray):
    """
    Plota o Método do Cotovelo (Inércia) e o Silhouette Score para K de 1 a 10,
    auxiliando na escolha do melhor número de clusters.
    """
    inercias = []
    silhuetas = []
    valores_k = range(2, 11)  # Silhouette exige K >= 2
    
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
        if k >= 2:
            silhuetas.append(silhouette_score(X, kmeans.labels_))
            
    # Gráfico 1: Método do Cotovelo (WCSS)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), inercias, marker='o', color='b', linestyle='-')
    plt.title('Método do Cotovelo (WCSS)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    
    # Gráfico 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(valores_k, silhuetas, marker='s', color='g', linestyle='-')
    plt.title('Silhouette Score por K')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Score de Silhueta')
    
    plt.tight_layout()
    plt.savefig('metodo_cotovelo_silueta.png')
    print("[INFO] Gráfico do Cotovelo e Silueta salvo como 'metodo_cotovelo_silueta.png'.")
    plt.show()


def plotar_clusters(X_2d: np.ndarray, y_predito: np.ndarray, centros_2d: np.ndarray = None):
    """
    Plota a dispersão dos dados agrupados pelo K-Means em 2D.
    """
    plt.figure(figsize=(8, 6))
    df_plot = pd.DataFrame(X_2d, columns=['Atributo 1', 'Atributo 2'])
    df_plot['Cluster'] = y_predito
    
    sns.scatterplot(
        x='Atributo 1', y='Atributo 2', hue='Cluster', 
        data=df_plot, palette='Set1', s=70, alpha=0.8, edgecolor='k'
    )
    
    if centros_2d is not None:
        plt.scatter(
            centros_2d[:, 0], centros_2d[:, 1], 
            s=250, c='yellow', marker='X', edgecolor='black', label='Centróides'
        )
        
    plt.title('Agrupamento K-Means (Visualização 2D)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('clusters_iris.png')
    print("[INFO] Gráfico de dispersão dos clusters salvo como 'clusters_iris.png'.")
    plt.show()


# ==========================================
# FLUXO PRINCIPAL DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # 1. Justificativa Sucinta de Atributos:
    # Utilizaremos todos os 4 atributos numéricos da base Iris (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm).
    # Justificativa: A base apresenta baixa dimensionalidade e alta relevância botânica em todas as colunas, 
    # dispensando redução de dimensionalidade por PCA para preservar a interpretabilidade geométrica direta.
    
    caminho_csv = "Iris.csv"
    
    try:
        df = carregar_dados(caminho_csv)
    except FileNotFoundError as e:
        print(f"[ERRO] {e}")
        exit()
        
    colunas_atributos = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = df[colunas_atributos].values
    y_real = df['Species'].values
    
    print(f"\n[INFO] Atributos selecionados: {colunas_atributos}")
    
    # 2. Entrada do Usuário para Pré-processamento (Com a Opção 0 inclusa)
    print("\nEscolha o método de pré-processamento dos atributos:")
    print("  [0] Sem pré-processamento (Dados brutos originais)")
    print("  [1] Normalização (MinMaxScaler - intervalo [0, 1])")
    print("  [2] Padronização (StandardScaler - Média 0 e Desvio Padrão 1)")
    try:
        opcao_prep = int(input("Digite sua opção (0, 1 ou 2): "))
    except ValueError:
        opcao_prep = 1
        print("[AVISO] Entrada inválida. Assumindo opção padrão: 1 (Normalização).")
        
    X_processado = pre_processar_dados(X, opcao_prep)
    
    # 3. Geração dos gráficos auxiliares (Método do Cotovelo e Silueta)
    print("\n[INFO] Gerando gráficos do Método do Cotovelo e Coeficiente de Silhueta...")
    metodo_cotovelo_e_silueta(X_processado)  # <--- Nome corrigido aqui
    
    # 4. Entrada do Usuário para definição de K
    try:
        k_usuario = int(input("\nDigite a quantidade de clusters (K) que deseja testar (ex: 3): "))
    except ValueError:
        k_usuario = 3
        print("[AVISO] Entrada inválida. Assumindo K = 3 por padrão.")
        
    # 5. Execução do Algoritmo K-Means com o K escolhido
    print(f"\n[INFO] Executando K-Means com K = {k_usuario}...")
    kmeans = KMeans(n_clusters=k_usuario, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_processado)
    
    # 6. Avaliação completa (WCSS, Silhouette, Acurácia, Precisão, Revocação e Matriz de Confusão)
    avaliar_agrupamentos_e_classificador(X_processado, y_real, y_pred, k_usuario, kmeans)
    
    # 7. Visualização Gráfica 2D dos Clusters (Usando os 2 primeiros atributos processados)
    plotar_clusters(X_processado[:, :2], y_pred, kmeans.cluster_centers_[:, :2])