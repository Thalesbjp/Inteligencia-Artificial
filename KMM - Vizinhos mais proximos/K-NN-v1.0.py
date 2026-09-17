import math
from collections import Counter
from typing import Dict, List, Tuple
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


class ScalerError(Exception):
    """Exceção customizada para erros de normalização."""

    pass


class StandardScalerCustom:
    """Aplica Z-score normalization para garantir igualdade de escala entre features."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerCustom":
        if X.size == 0:
            raise ScalerError("Conjunto de dados vazio fornecido para fit.")
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Substitui desvio padrão zero por 1 para evitar divisão por zero
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ScalerError("O scaler precisa ser ajustado (fit) antes do transform.")
        if X.shape[1] != len(self.mean_):
            raise ScalerError("Dimensão de dados incompatível com os parâmetros treinados.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class KNNClassifierFromScratch:
    """Implementação limpa e modular do algoritmo K-Nearest Neighbors sem bibliotecas externas no modelo."""

    def __init__(self, k: int = 3) -> None:
        if k < 1:
            raise ValueError("O valor de k deve ser um número inteiro estritamente positivo (k >= 1).")
        self.k: int = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Os dados de treino não podem estar vazios.")
        if len(X) != len(y):
            raise ValueError("X e y devem possuir o mesmo número de amostras.")
        if self.k > len(X):
            raise ValueError(f"O valor de k ({self.k}) não pode ser maior que o número de amostras de treino ({len(X)}).")

        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)

    def _euclidean_distance(self, row1: np.ndarray, row2: np.ndarray) -> float:
        """Calcula a Distância Euclidiana entre dois vetores contínuos."""
        squared_sum = sum((a - b) ** 2 for a, b in zip(row1, row2))
        return math.sqrt(squared_sum)

    def _predict_single(self, x_test_row: np.ndarray) -> int:
        """Determina a classe de um único ponto baseando-se no voto de maioria dos vizinhos."""
        distances: List[Tuple[float, int]] = []

        # Calcula a distância de uma única amostra para todas as amostras de treino
        for x_train_row, y_label in zip(self.X_train, self.y_train):
            dist = self._euclidean_distance(x_test_row, x_train_row)
            distances.append((dist, int(y_label)))

        # Ordena as distâncias de forma crescente e obtém as rótulos dos k mais próximos
        distances.sort(key=lambda item: item[0])
        k_nearest_labels = [label for _, label in distances[: self.k]]

        # Em caso de empate na contagem, Counter.most_common retorna a primeira classe que atingiu o pico
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("O modelo precisa ser treinado (fit) antes de realizar predições.")

        X_test_arr = np.asarray(X_test, dtype=np.float64)
        if X_test_arr.ndim == 1:
            X_test_arr = X_test_arr.reshape(1, -1)

        if X_test_arr.shape[1] != self.X_train.shape[1]:
            raise ValueError("O número de atributos de entrada difere dos dados de treino.")

        predictions = [self._predict_single(row) for row in X_test_arr]
        return np.array(predictions)


class IrisPipeline:
    """Gerenciador do pipeline de dados, avaliação de hiperparâmetros e relatórios."""

    def __init__(self, test_size: float = 0.3, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScalerCustom()
        self.target_names: List[str] = []
        self.feature_names: List[str] = []

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega o dataset Iris do Scikit-Learn e realiza o split com normalização."""
        iris_data = load_iris()
        self.target_names = list(iris_data.target_names)
        self.feature_names = list(iris_data.feature_names)

        df = pd.DataFrame(data=iris_data.data, columns=self.feature_names)
        X = df.values
        y = iris_data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # O scaler é ajustado SOMENTE nos dados de treino para evitar Vazamento de Dados (Data Leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate_k_values(
        self,
        k_list: List[int],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[int, float]:
        """Avalia a taxa de acurácia do modelo para uma lista de valores k."""
        results = {}
        for k in k_list:
            knn = KNNClassifierFromScratch(k=k)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            results[k] = float(acc)
        return results

    def display_metrics_and_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, k_value: int) -> None:
        """Gera relatório textual resumido e plota a Matriz de Confusão para um dado K."""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)

        print(f"\n==========================================")
        print(f"    MÉTRICAS DETALHADAS PARA K = {k_value}")
        print(f"==========================================")
        print(f"Acurácia : {acc * 100:.2f}%")
        print(f"Precisão : {precision * 100:.2f}% (Macro)")
        print(f"Revocação: {recall * 100:.2f}% (Macro)")
        print(f"F1-Score : {f1 * 100:.2f}% (Macro)")

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão (K={k_value})")
        plt.tight_layout()
        plt.show()


# =====================================================================
# BLOCO DE EXECUÇÃO E TESTE PRÁTICO
# =====================================================================
if __name__ == "__main__":
    try:
        pipeline = IrisPipeline(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = pipeline.load_and_prepare_data()

        # 1. Testando taxas de reconhecimento para k = {1, 3, 5, 7}
        k_candidates = [1, 3, 5, 7]
        accuracies = pipeline.evaluate_k_values(k_candidates, X_train, X_test, y_train, y_test)

        print("--- TAXAS DE RECONHECIMENTO ---")
        for k_val, acc in accuracies.items():
            print(f"Taxa de reconhecimento para k={k_val}: {acc * 100:.2f}%")

        # 2. Exibindo matriz de confusão e métricas completas para k=3
        best_k = 3
        model = KNNClassifierFromScratch(k=best_k)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        pipeline.display_metrics_and_matrix(y_test, y_preds, k_value=best_k)

        # 3. Exemplo Prático: Classificando uma nova amostra de entrada
        # [Sepal Length, Sepal Width, Petal Length, Petal Width]
        raw_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

        # A amostra DEVE passar pela mesma escala Z-score aplicada no treino
        scaled_sample = pipeline.scaler.transform(raw_sample)
        predicted_class_idx = model.predict(scaled_sample)[0]
        predicted_class_name = pipeline.target_names[predicted_class_idx]

        print("\n--- CLASSIFICAÇÃO DE NOVA ENTRADA ---")
        print(f"Atributos originais da amostra: {raw_sample[0]}")
        print(f"Classe Predita: {predicted_class_name.upper()} (Índice: {predicted_class_idx})")

    except Exception as e:
        print(f"\n[ERRO DE EXECUÇÃO]: {e}")