"""
TRABALHO PRÁTICO 02 - K-MEANS
Base: Iris.csv
"""

from pathlib import Path

# ============================================================
# 1. PARÂMETROS PRINCIPAIS DO EXPERIMENTO
# ============================================================
# K: quantidade de grupos (clusters) que o K-Means deverá encontrar.
K = 7


# TRANSFORMACAO:
#   1 = Normalização (coloca os atributos, aproximadamente, entre 0 e 1)
#   2 = Padronização (média 0 e desvio-padrão 1)
TRANSFORMACAO = 2

# ATRIBUTOS:
# Quantidade de atributos utilizados na classificação/agrupamento.
# Neste trabalho serão utilizados os 4: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
ATRIBUTOS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

# Faixa de K usada somente para analisar o Método do Cotovelo
# e o Silhouette Score. O K escolhido acima continua sendo o K principal.
K_MIN = 2
K_MAX = 8

# Nome do arquivo da base. Deve estar na mesma pasta deste programa.
ARQUIVO = "Iris.csv"

# Caminho da pasta onde este arquivo .py está salvo.
PASTA_PROGRAMA = Path(__file__).resolve().parent
CAMINHO_ARQUIVO = PASTA_PROGRAMA / ARQUIVO


# ============================================================
# 2. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.decomposition import PCA
from itertools import permutations


# ============================================================
# 3. FUNÇÕES DE TRATAMENTO DOS DADOS
# ============================================================

def carregar_dados(nome_arquivo):
    try:
        dados = pd.read_csv(nome_arquivo)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\nArquivo '{nome_arquivo}' não encontrado.\n"
            "Coloque o Iris.csv na mesma pasta deste programa."
        )

    return dados


def preparar_atributos(dados, atributos, tipo_transformacao):
    X = dados[atributos].copy()

    if tipo_transformacao == 1:
        # MinMaxScaler faz a normalização dos valores.
        transformador = MinMaxScaler()
        nome = "Normalização"

    elif tipo_transformacao == 2:
        # StandardScaler faz a padronização:
        # média aproximadamente 0 e desvio-padrão aproximadamente 1.
        transformador = StandardScaler()
        nome = "Padronização"

    else:
        raise ValueError("A transformação deve ser 1 (normalização) ou 2 (padronização).")

    X_transformado = transformador.fit_transform(X)

    print(f"\nTransformação escolhida: {nome}")
    print(f"Quantidade de atributos: {len(atributos)}")
    print(f"Atributos utilizados: {', '.join(atributos)}")

    return X_transformado


# ============================================================
# 4. EXECUÇÃO DO K-MEANS
# ============================================================

def executar_kmeans(X, k):
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # labels_ contém o número do cluster de cada flor.
    clusters = modelo.fit_predict(X)

    return modelo, clusters


# ============================================================
# 5. MÉTODO DO COTOVELO E SILHOUETTE
# ============================================================

def analisar_valores_de_k(X, k_min, k_max):
    valores_k = []
    inercia = []
    silhueta = []

    print("\n" + "=" * 60)
    print("ANÁLISE DE DIFERENTES VALORES DE K")
    print("=" * 60)

    for k in range(k_min, k_max + 1):
        modelo, clusters = executar_kmeans(X, k)

        valores_k.append(k)
        inercia.append(modelo.inertia_)

        # O Silhouette exige pelo menos 2 clusters e menos clusters
        # que o número de amostras.
        silhueta.append(silhouette_score(X, clusters))

        print(
            f"K = {k:2d} | "
            f"WCSS/Inércia = {modelo.inertia_:8.3f} | "
            f"Silhouette = {silhueta[-1]:.3f}"
        )

    # ---------------- Método do Cotovelo ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, inercia, marker="o")
    plt.title("Método do Cotovelo - K-Means")
    plt.xlabel("Quantidade de clusters (K)")
    plt.ylabel("Inércia / WCSS")
    plt.xticks(valores_k)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------- Silhouette ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, silhueta, marker="o")
    plt.title("Silhouette Score por valor de K")
    plt.xlabel("Quantidade de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(valores_k)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return valores_k, inercia, silhueta


# ============================================================
# 6. VISUALIZAÇÃO DOS CLUSTERS COM PCA
# ============================================================

def plotar_clusters_pca(X, clusters, k):
    pca = PCA(n_components=2)

    # X_pca possui duas dimensões: componente 1 e componente 2.
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    # Cada número representa o cluster encontrado pelo K-Means.
    for cluster in range(k):
        pontos = X_pca[clusters == cluster]

        plt.scatter(
            pontos[:, 0],
            pontos[:, 1],
            label=f"Cluster {cluster}"
        )

    # Os centroides também são transformados para o espaço do PCA.
    # Isso permite desenhá-los junto com os pontos.
    modelo_auxiliar = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    ).fit(X)

    centroides_pca = pca.transform(modelo_auxiliar.cluster_centers_)

    plt.scatter(
        centroides_pca[:, 0],
        centroides_pca[:, 1],
        marker="X",
        s=180,
        label="Centroides"
    )

    plt.title("Agrupamentos encontrados pelo K-Means - PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. MATRIZ DE CONFUSÃO
# ============================================================

def alinhar_clusters_com_classes(clusters, classes_reais):
    """
    Alinha os números dos clusters às classes reais.

    Por que isso é necessário?
        K-Means é NÃO supervisionado. Portanto, o Cluster 0 não significa
        obrigatoriamente Iris-setosa, por exemplo.

        Como a avaliação abaixo é feita com K=3, testamos as possíveis
        correspondências entre os 3 clusters e as 3 espécies e escolhemos
        a que gera o maior número de acertos.

    Observação:
        Species é usada SOMENTE depois do agrupamento para comparação.
        Ela nunca entra como atributo do K-Means.
    """
    classes = np.unique(classes_reais)
    clusters_unicos = np.unique(clusters)

    melhor_mapeamento = None
    maior_numero_acertos = -1

    # Para K=3, existem 3! = 6 formas de relacionar clusters e classes.
    # Testamos todas e escolhemos a que produz mais correspondências.
    for permutacao in permutations(classes):
        mapeamento_teste = dict(zip(clusters_unicos, permutacao))

        previsoes_teste = np.array([
            mapeamento_teste[cluster]
            for cluster in clusters
        ])

        acertos = np.sum(previsoes_teste == classes_reais)

        if acertos > maior_numero_acertos:
            maior_numero_acertos = acertos
            melhor_mapeamento = mapeamento_teste

    previsoes = np.array([
        melhor_mapeamento[cluster]
        for cluster in clusters
    ])

    return previsoes, melhor_mapeamento


def avaliar_com_classes(dados, clusters, k):
    """
    Calcula matriz de confusão, precisão, revocação e acurácia.

    Atenção:
        Essas métricas são uma COMPARAÇÃO entre os clusters encontrados
        e os rótulos conhecidos da Iris. Elas não mudam o fato de que
        K-Means é um algoritmo não supervisionado.

        Como a Iris possui 3 espécies, a avaliação de classificação
        fica conceitualmente mais direta quando K = 3.
    """
    classes_reais = dados["Species"].values

    if k != len(np.unique(classes_reais)):
        print("\n" + "=" * 60)
        print("AVALIAÇÃO COM RÓTULOS")
        print("=" * 60)
        print(
            f"K = {k}, mas a base possui {len(np.unique(classes_reais))} espécies."
        )
        print(
            "A matriz de confusão será apresentada como comparação entre "
            "espécies e clusters. As métricas de classificação "
            "(precisão, revocação e acurácia) não serão calculadas, "
            "pois K diferente da quantidade de classes não representa "
            "uma classificação 1 para 1."
        )

        classes = np.unique(classes_reais)
        matriz = confusion_matrix(classes_reais, clusters, labels=classes)

        print("\nMatriz de confusão (linhas = espécie / colunas = cluster):")
        print(pd.DataFrame(
            matriz,
            index=classes,
            columns=[f"Cluster {i}" for i in range(k)]
        ))

        return

    # Quando K=3, podemos fazer a correspondência cluster <-> espécie.
    previsoes, mapeamento = alinhar_clusters_com_classes(
        clusters,
        classes_reais
    )

    classes = np.unique(classes_reais)

    matriz = confusion_matrix(
        classes_reais,
        previsoes,
        labels=classes
    )

    acuracia = accuracy_score(classes_reais, previsoes)

    # average="macro" calcula a métrica para cada classe e depois
    # tira a média, dando o mesmo peso para cada espécie.
    precisao = precision_score(
        classes_reais,
        previsoes,
        labels=classes,
        average="macro",
        zero_division=0
    )

    revocacao = recall_score(
        classes_reais,
        previsoes,
        labels=classes,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSÃO E MÉTRICAS")
    print("=" * 60)

    print("\nMapeamento encontrado:")
    for cluster, especie in mapeamento.items():
        print(f"Cluster {cluster} -> {especie}")

    print("\nMatriz de confusão:")
    print(pd.DataFrame(
        matriz,
        index=classes,
        columns=classes
    ))

    print(f"\nPrecisão (macro):   {precisao:.4f}")
    print(f"Revocação (macro): {revocacao:.4f}")
    print(f"Acurácia:           {acuracia:.4f}")

    # Gráfico simples da matriz de confusão.
    plt.figure(figsize=(7, 6))
    plt.imshow(matriz)

    plt.title("Matriz de Confusão - K-Means")
    plt.xlabel("Classe prevista após alinhamento")
    plt.ylabel("Classe real")

    plt.xticks(range(len(classes)), classes, rotation=30)
    plt.yticks(range(len(classes)), classes)

    # Escreve o valor dentro de cada célula.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, matriz[i, j], ha="center", va="center")

    plt.colorbar(label="Quantidade de amostras")
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. PROGRAMA PRINCIPAL
# ============================================================

def main():
    """
    Função principal:
        1. Carrega a base;
        2. Seleciona os atributos;
        3. Normaliza/padroniza;
        4. Analisa diferentes K;
        5. Executa o K escolhido;
        6. Calcula Silhouette;
        7. Mostra os clusters com PCA;
        8. Compara os clusters com as espécies conhecidas.
    """

    print("=" * 60)
    print("K-MEANS - BASE IRIS")
    print("=" * 60)

    print(f"K escolhido: {K}")
    print(
        "Transformação:",
        "Normalização" if TRANSFORMACAO == 1 else "Padronização"
    )
    print(f"Quantidade de atributos: {len(ATRIBUTOS)}")
    print(f"Atributos: {', '.join(ATRIBUTOS)}")

    # --------------------------------------------------------
    # Carregamento
    # --------------------------------------------------------
    dados = carregar_dados(CAMINHO_ARQUIVO)

    print(f"\nQuantidade de amostras: {len(dados)}")

    # --------------------------------------------------------
    # Preparação dos dados
    # --------------------------------------------------------
    X = preparar_atributos(
        dados,
        ATRIBUTOS,
        TRANSFORMACAO
    )

    # --------------------------------------------------------
    # Análise de diferentes valores de K
    # --------------------------------------------------------
    analisar_valores_de_k(
        X,
        K_MIN,
        K_MAX
    )

    # --------------------------------------------------------
    # Execução final com o K definido no início
    # --------------------------------------------------------
    modelo, clusters = executar_kmeans(X, K)

    print("\n" + "=" * 60)
    print("RESULTADO DO K-MEANS")
    print("=" * 60)

    print(f"K utilizado: {K}")
    print(f"WCSS / Inércia: {modelo.inertia_:.4f}")

    silhueta_final = silhouette_score(X, clusters)
    print(f"Silhouette Score: {silhueta_final:.4f}")

    # --------------------------------------------------------
    # Visualização dos agrupamentos
    # --------------------------------------------------------
    plotar_clusters_pca(
        X,
        clusters,
        K
    )

    # --------------------------------------------------------
    # Comparação com as espécies conhecidas
    # --------------------------------------------------------
    avaliar_com_classes(
        dados,
        clusters,
        K
    )

    print("\nPrograma finalizado.")


# Esta condição faz com que main() seja executada somente
# quando este arquivo for executado diretamente.
if __name__ == "__main__":
    main()