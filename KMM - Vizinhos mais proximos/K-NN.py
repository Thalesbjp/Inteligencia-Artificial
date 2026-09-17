import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


class ErroNormalizacao(Exception):
    """Exceção customizada para erros de normalização."""

    pass


class NormalizadorPadrao:
    """Aplica normalização Z-score para garantir igualdade de escala entre atributos."""

    def __init__(self) -> None:
        self.media: np.ndarray | None = None
        self.desvio_padrao: np.ndarray | None = None

    def ajustar(self, dados: np.ndarray) -> "NormalizadorPadrao":
        if dados.size == 0:
            raise ErroNormalizacao("Conjunto de dados vazio fornecido para ajuste.")
        self.media = np.mean(dados, axis=0)
        self.desvio_padrao = np.std(dados, axis=0)
        # Substitui desvio padrão zero por 1 para evitar divisão por zero
        self.desvio_padrao[self.desvio_padrao == 0.0] = 1.0
        return self

    def transformar(self, dados: np.ndarray) -> np.ndarray:
        if self.media is None or self.desvio_padrao is None:
            raise ErroNormalizacao("O normalizador precisa ser ajustado antes da transformação.")
        if dados.shape[1] != len(self.media):
            raise ErroNormalizacao("Dimensão de dados incompatível com os parâmetros ajustados.")
        return (dados - self.media) / self.desvio_padrao

    def ajustar_transformar(self, dados: np.ndarray) -> np.ndarray:
        return self.ajustar(dados).transformar(dados)


class ClassificadorKNN:
    """Implementação limpa e modular do algoritmo K vizinhos mais próximos."""

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("O valor de k deve ser um número inteiro estritamente positivo (k >= 1).")
        self.k: int = k
        self.dados_treino: np.ndarray | None = None
        self.classes_treino: np.ndarray | None = None

    def ajustar(self, dados: np.ndarray, classes: np.ndarray) -> None:
        if len(dados) == 0 or len(classes) == 0:
            raise ValueError("Os dados de treino não podem estar vazios.")
        if len(dados) != len(classes):
            raise ValueError("Os dados e as classes devem possuir o mesmo número de amostras.")
        if self.k > len(dados):
            raise ValueError(f"O valor de k ({self.k}) não pode ser maior que o número de amostras de treino ({len(dados)}).")

        self.dados_treino = np.asarray(dados, dtype=np.float64)
        self.classes_treino = np.asarray(classes)

    def _calcular_distancia_euclidiana(self, linha1: np.ndarray, linha2: np.ndarray) -> float:
        """Calcula a Distância Euclidiana entre dois vetores contínuos."""
        soma_quadrados = sum((valor1 - valor2) ** 2 for valor1, valor2 in zip(linha1, linha2))
        return math.sqrt(soma_quadrados)

    def _prever_unico(self, linha_teste: np.ndarray) -> int:
        """Determina a classe de um único ponto baseando-se no voto de maioria dos vizinhos."""
        distancias: list[tuple[float, int]] = []

        # Calcula a distância de uma única amostra para todas as amostras de treino
        for linha_treino, rotulo in zip(self.dados_treino, self.classes_treino):
            distancia = self._calcular_distancia_euclidiana(linha_teste, linha_treino)
            distancias.append((distancia, int(rotulo)))

        # Ordena as distâncias e obtém os rótulos dos k vizinhos mais próximos
        distancias.sort(key=lambda item: item[0])
        rotulos_mais_proximos = [rotulo for _, rotulo in distancias[: self.k]]

        # Em caso de empate, Counter.most_common retorna a primeira classe mais frequente
        rotulo_mais_frequente = Counter(rotulos_mais_proximos).most_common(1)
        return rotulo_mais_frequente[0][0]

    def prever(self, dados_teste: np.ndarray) -> np.ndarray:
        if self.dados_treino is None or self.classes_treino is None:
            raise RuntimeError("O modelo precisa ser ajustado antes de realizar previsões.")

        dados_teste_array = np.asarray(dados_teste, dtype=np.float64)
        if dados_teste_array.ndim == 1:
            dados_teste_array = dados_teste_array.reshape(1, -1)

        if dados_teste_array.shape[1] != self.dados_treino.shape[1]:
            raise ValueError("O número de atributos de entrada difere dos dados de treino.")

        previsoes = [self._prever_unico(linha) for linha in dados_teste_array]
        return np.array(previsoes)


class PipelineIris:
    """Gerenciador do pipeline de dados, avaliação de hiperparâmetros e relatórios."""

    def __init__(self, tamanho_teste: float = 0.3, estado_aleatorio: int = 42) -> None:
        self.tamanho_teste = tamanho_teste
        self.estado_aleatorio = estado_aleatorio
        self.normalizador = NormalizadorPadrao()
        self.nomes_alvo: list[str] = []
        self.nomes_atributos: list[str] = []

    def carregar_e_preparar_dados(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega o conjunto de dados Iris do Scikit-Learn e realiza a divisão com normalização."""
        dados_iris = load_iris()
        self.nomes_alvo = list(dados_iris.target_names)
        self.nomes_atributos = list(dados_iris.feature_names)

        tabela = pd.DataFrame(data=dados_iris.data, columns=self.nomes_atributos)
        atributos = tabela.values
        classes = dados_iris.target

        atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
            atributos, classes, test_size=self.tamanho_teste,
            random_state=self.estado_aleatorio, stratify=classes
        )

        # O normalizador é ajustado somente nos dados de treino para evitar vazamento de dados
        atributos_treino_normalizados = self.normalizador.ajustar_transformar(atributos_treino)
        atributos_teste_normalizados = self.normalizador.transformar(atributos_teste)

        return atributos_treino_normalizados, atributos_teste_normalizados, classes_treino, classes_teste

    def avaliar_valores_k(
        self,
        lista_k: list[int],
        atributos_treino: np.ndarray,
        atributos_teste: np.ndarray,
        classes_treino: np.ndarray,
        classes_teste: np.ndarray,
    ) -> dict[int, float]:
        """Avalia a taxa de acurácia do modelo para uma lista de valores k."""
        resultados = {}
        for k in lista_k:
            knn = ClassificadorKNN(k=k)
            knn.ajustar(atributos_treino, classes_treino)
            previsoes = knn.prever(atributos_teste)
            acuracia = accuracy_score(classes_teste, previsoes)
            resultados[k] = float(acuracia)
        return resultados

    def exibir_metricas_e_matriz(self, classes_reais: np.ndarray, classes_previstas: np.ndarray, valor_k: int) -> None:
        """Gera relatório textual resumido e plota a Matriz de Confusão para um dado K."""
        precisao, revocacao, medida_f1, _ = precision_recall_fscore_support(classes_reais, classes_previstas, average="macro")
        acuracia = accuracy_score(classes_reais, classes_previstas)

        print(f"\n==========================================")
        print(f"    MÉTRICAS DETALHADAS PARA K = {valor_k}")
        print(f"==========================================")
        print(f"Acurácia : {acuracia * 100:.2f}%")
        print(f"Precisão : {precisao * 100:.2f}% (Macro)")
        print(f"Revocação: {revocacao * 100:.2f}% (Macro)")
        print(f"Pontuação F1: {medida_f1 * 100:.2f}% (Macro)")

        matriz_confusao = confusion_matrix(classes_reais, classes_previstas)
        exibicao = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=self.nomes_alvo)
        exibicao.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão (K={valor_k})")
        plt.tight_layout()
        plt.show()


# =====================================================================
# BLOCO DE EXECUÇÃO E TESTE PRÁTICO
# =====================================================================
if __name__ == "__main__":
    try:
        pipeline = PipelineIris(tamanho_teste=0.3, estado_aleatorio=42)
        atributos_treino, atributos_teste, classes_treino, classes_teste = pipeline.carregar_e_preparar_dados()

        # 1. Testando taxas de reconhecimento para k = {1, 3, 5, 7}
        candidatos_k = [1, 3, 5, 7]
        acuracias = pipeline.avaliar_valores_k(
            candidatos_k, atributos_treino, atributos_teste, classes_treino, classes_teste
        )

        print("--- TAXAS DE RECONHECIMENTO ---")
        for valor_k, acuracia in acuracias.items():
            print(f"Taxa de reconhecimento para k={valor_k}: {acuracia * 100:.2f}%")

        # 2. Exibindo matriz de confusão e métricas completas para k=3
        melhor_k = 3
        modelo = ClassificadorKNN(k=melhor_k)
        modelo.ajustar(atributos_treino, classes_treino)
        classes_previstas = modelo.prever(atributos_teste)

        pipeline.exibir_metricas_e_matriz(classes_teste, classes_previstas, valor_k=melhor_k)

        # 3. Exemplo Prático: Classificando uma nova amostra de entrada
        # [Sepal Length, Sepal Width, Petal Length, Petal Width]
        amostra_original = np.array([[5.1, 3.5, 1.4, 0.2]])

        # A amostra DEVE passar pela mesma escala Z-score aplicada no treino
        amostra_normalizada = pipeline.normalizador.transformar(amostra_original)
        indice_classe_prevista = modelo.prever(amostra_normalizada)[0]
        nome_classe_prevista = pipeline.nomes_alvo[indice_classe_prevista]

        print("\n--- CLASSIFICAÇÃO DE NOVA ENTRADA ---")
        print(f"Atributos originais da amostra: {amostra_original[0]}")
        print(f"Classe Predita: {nome_classe_prevista.upper()} (Índice: {indice_classe_prevista})")

    except Exception as e:
        print(f"\n[ERRO DE EXECUÇÃO]: {e}")