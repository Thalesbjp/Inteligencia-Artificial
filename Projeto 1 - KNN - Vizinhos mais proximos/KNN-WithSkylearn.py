import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# --------- Configurações ---------
K_LIST = [1, 3, 5, 7]
CSV_PATH = "Iris.csv"

CLASS_COL_CANDIDATES = ["species", "Species", "class", "Class", "target", "Target"]

# --------- Funções ---------
def plot_confusion(cm, labels, k):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Matriz de Confusão (sklearn K={k})")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.show()

# --------- Main ---------
df = pd.read_csv(CSV_PATH)

class_col = None
for c in df.columns:
    if c in CLASS_COL_CANDIDATES:
        class_col = c
        break
if class_col is None:
    class_col = df.columns[-1]

X = df.drop(columns=[class_col]).values.astype(float)
y_raw = df[class_col].values

labels_names = sorted(list(set(y_raw)))
label_to_int = {lab: i for i, lab in enumerate(labels_names)}
y = np.array([label_to_int[v] for v in y_raw], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []

for k in K_LIST:
    t0 = time.time()

    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    t1 = time.time()

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels_names))))
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    results.append((k, precision_macro, recall_macro, accuracy, t1 - t0))

    print(f"\n=== K={k} (sklearn) ===")
    print(f"Acurácia:  {accuracy:.4f}")
    print(f"Precisão (macro): {precision_macro:.4f}")
    print(f"Revocação (macro): {recall_macro:.4f}")
    print(f"Tempo (s): {t1 - t0:.6f}")

    plot_confusion(cm, labels_names, k)

print("\nResumo geral (sklearn):")
for k, p, r, a, t in results:
    print(f"K={k} | Acc={a:.4f} | Prec(macro)={p:.4f} | Rec(macro)={r:.4f} | Tempo={t:.6f}s") 