import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter

# --------- Configurações ---------
K_LIST = [1, 3, 5, 7]
CSV_PATH = "Iris.csv"

# Para padronizar (você pode ajustar conforme o nome da coluna)
CLASS_COL_CANDIDATES = ["species", "Species", "class", "Class", "target", "Target"]

# --------- Funções ---------
def train_test_split_simple(X, y, test_ratio=0.5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    test_size = int(len(X) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def euclidean_dist(a, b):
    # a e b: vetores 1D
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict_one(train_X, train_y, x, k):
    # calcula distâncias
    dists = []
    for i in range(len(train_X)):
        d = euclidean_dist(x, train_X[i])
        dists.append((d, train_y[i]))
    dists.sort(key=lambda t: t[0])

    # pega k vizinhos mais próximos
    k_neigh = dists[:k]
    labels = [lab for _, lab in k_neigh]

    # voto majoritário
    counts = Counter(labels)
    max_votes = max(counts.values())
    top_labels = [lab for lab, c in counts.items() if c == max_votes]

    # desempate: pega o mais comum entre os empatados com menor distância média (simples)
    if len(top_labels) == 1:
        return top_labels[0]
    else:
        avg_dist = {}
        for lab in top_labels:
            dist_list = [d for d, yv in k_neigh if yv == lab]
            avg_dist[lab] = float(np.mean(dist_list))
        # menor distância média vence
        return min(avg_dist, key=avg_dist.get)

def confusion_matrix_manual(y_true, y_pred, labels):
    m = np.zeros((len(labels), len(labels)), dtype=int)
    idx_map = {lab: i for i, lab in enumerate(labels)}
    for yt, yp in zip(y_true, y_pred):
        m[idx_map[yt], idx_map[yp]] += 1
    return m

def precision_recall_accuracy(y_true, y_pred, labels):
    # métricas macro: calcula por classe e tira média
    cm = confusion_matrix_manual(y_true, y_pred, labels)

    precisions = []
    revocacoes = []

    for i in range(len(labels)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(prec)
        revocacoes.append(rec)

    precision_macro = float(np.mean(precisions))
    recall_macro = float(np.mean(revocacoes))
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

    return precision_macro, recall_macro, accuracy

def plot_confusion(cm, labels, k):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Matriz de Confusão (K={k})")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    # valores no heatmap
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.show()

# --------- Main ---------
df = pd.read_csv(CSV_PATH)

# Descobrir coluna da classe
class_col = None
for c in df.columns:
    if c in CLASS_COL_CANDIDATES:
        class_col = c
        break

# Se não achou, tenta heurística: última coluna
if class_col is None:
    class_col = df.columns[-1]

# Características: assumimos as colunas exceto classe
X_df = df.drop(columns=[class_col])
y_raw = df[class_col].values

# Converte para números (opcional, mas facilita labels consistentes)
labels = sorted(list(set(y_raw)))
label_to_int = {lab: i for i, lab in enumerate(labels)}
y = np.array([label_to_int[v] for v in y_raw], dtype=int)

X = X_df.values.astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_ratio=0.2, seed=42)

# Converte labels para exibir nomes originais
labels_int_to_name = {label_to_int[lab]: lab for lab in labels}
labels_names_ordered = [labels_int_to_name[i] for i in range(len(labels))]

results = []
 
for k in K_LIST:
    t0 = time.time()

    y_pred = []
    for x in X_test:
        pred = knn_predict_one(X_train, y_train, x, k)
        y_pred.append(pred)
    y_pred = np.array(y_pred, dtype=int)

    precision_macro, recall_macro, accuracy = precision_recall_accuracy(
        y_test, y_pred, labels=list(range(len(labels)))
    )

    t1 = time.time()
    cm = confusion_matrix_manual(y_test, y_pred, labels=list(range(len(labels))))

    results.append((k, precision_macro, recall_macro, accuracy, t1 - t0))

    print(f"\n=== K={k} ===")
    print(f"Acurácia:  {accuracy:.4f}")
    print(f"Precisão (macro): {precision_macro:.4f}")
    print(f"Revocação (macro): {recall_macro:.4f}")
    print(f"Tempo (s): {t1 - t0:.6f}")

    plot_confusion(cm, labels_names_ordered, k)

# Resumo final
print("\nResumo geral:")
for k, p, r, a, t in results:
    print(f"K={k} | Acc={a:.4f} | Prec(macro)={p:.4f} | Rec(macro)={r:.4f} | Tempo={t:.6f}s")