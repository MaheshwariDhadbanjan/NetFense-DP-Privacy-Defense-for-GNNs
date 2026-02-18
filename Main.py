import threading
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tkinter import filedialog, messagebox, Text, Tk, Button, END
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dp_layer import add_dp_noise  # Ensure you have this function in dp_layer.py

# GUI setup
main = Tk()
main.title("NetFense: Adversarial Defenses with DP")
main.geometry("1300x900")

text = Text(main, height=30, width=150)
text.place(x=50, y=130)

# Globals
nodes, labels, source, target, edges, perturbed_df = None, None, None, None, None, None

def uploadDataset():
    global nodes, labels, source, target, edges
    text.delete('1.0', END)
    filedialog.askdirectory()

    nodes = pd.read_csv("Dataset/nodes.csv")
    edges = pd.read_csv("Dataset/edges.csv")
    source = edges['source'].values
    target = edges['target'].values
    labels = nodes['label'].values

    text.insert(END, "📂 Dataset Loaded:\n\n")
    text.insert(END, "📘 Nodes Dataset (First 10 rows):\n")
    text.insert(END, str(nodes.head(10)) + "\n\n")
    text.insert(END, "📗 Edges Dataset (First 10 rows):\n")
    text.insert(END, str(edges.head(10)) + "\n\n")
    text.insert(END, f"🔹 Total Edges in Original Graph: {edges.shape[0]}\n")

def getLabel(src):
    for i in range(len(nodes)):
        if nodes['nodeID'][i] == src:
            return labels[i]
    return 0

def purturbedData():
    def task():
        global perturbed_df
        text.delete('1.0', END)
        text.insert(END, "\n⚙️ Generating Perturbed Data with DP...\n")

        perturbed = []
        if source is None or target is None:
            text.insert(END, "❌ Upload dataset first.\n")
            return

        random_edges = np.random.randint(low=1, high=np.max(source), size=30)

        for i in range(len(target)):
            if target[i] not in random_edges:
                perturbed.append([source[i], target[i], getLabel(source[i])])

        random_source = np.random.randint(low=1, high=np.max(source), size=80)
        random_target = np.random.randint(low=1, high=np.max(target), size=80)

        for i in range(len(random_source)):
            perturbed.append([random_source[i], random_target[i], getLabel(random_source[i])])

        perturbed_df = pd.DataFrame(perturbed, columns=['source', 'edge', 'label'])
        features = perturbed_df[['source', 'edge']].values
        features_dp = add_dp_noise(features, epsilon=1.0)
        perturbed_df[['source', 'edge']] = pd.DataFrame(features_dp.astype(int))

        perturbed_df.to_csv("perturbed.csv", index=False)

        text.insert(END, "\n🧾 Perturbed Dataset (First 10 rows):\n")
        text.insert(END, str(perturbed_df.head(10)) + "\n")
        text.insert(END, f"\n🔹 Total Edges in Perturbed Graph: {perturbed_df.shape[0]}\n")

    threading.Thread(target=task).start()

def compareGraphs():
    def task():
        if perturbed_df is None or edges is None:
            messagebox.showwarning("Missing Data", "Please upload dataset and generate perturbed data.")
            return

        text.delete('1.0', END)
        text.insert(END, "\n🔍 Comparing Graph Structure & Model Accuracy...\n")

        G_orig = nx.Graph()
        for _, row in edges.iterrows():
            G_orig.add_edge(row['source'], row['target'])

        G_pert = nx.Graph()
        for _, row in perturbed_df.iterrows():
            G_pert.add_edge(row['source'], row['edge'])

        d_orig = nx.density(G_orig)
        d_pert = nx.density(G_pert)
        cc_orig = nx.average_clustering(G_orig)
        cc_pert = nx.average_clustering(G_pert)

        text.insert(END, "\n📊 Graph Metrics:\n")
        text.insert(END, f"• Density (Original): {d_orig:.4f}\n")
        text.insert(END, f"• Density (Perturbed): {d_pert:.4f}\n")
        text.insert(END, f"• Clustering Coeff. (Original): {cc_orig:.4f}\n")
        text.insert(END, f"• Clustering Coeff. (Perturbed): {cc_pert:.4f}\n")

        # Model accuracy
        X_orig = edges[['source', 'target']].values
        Y_orig = [getLabel(s) for s in edges['source']]
        X_train, X_test, y_train, y_test = train_test_split(X_orig, Y_orig, test_size=0.2, random_state=42)
        acc_orig = accuracy_score(y_test, RandomForestClassifier().fit(X_train, y_train).predict(X_test))

        X_pert = perturbed_df[['source', 'edge']].values
        Y_pert = perturbed_df['label'].values
        X_pert_dp = add_dp_noise(X_pert, epsilon=1.0)
        X_train_dp, X_test_dp, y_train_dp, y_test_dp = train_test_split(X_pert_dp, Y_pert, test_size=0.2, random_state=42)
        acc_dp = accuracy_score(y_test_dp, RandomForestClassifier().fit(X_train_dp, y_train_dp).predict(X_test_dp))

        text.insert(END, "\n🤖 Model Accuracy:\n")
        text.insert(END, f"• Accuracy (Original): {acc_orig:.2f}\n")
        text.insert(END, f"• Accuracy (With DP):  {acc_dp:.2f}\n")
        text.insert(END, f"\n🛡️ Estimated Privacy Gain: {acc_orig - acc_dp:.4f}\n")

        # Bar graph
        labels_bar = ['Density', 'Clustering Coeff.', 'Model Accuracy']
        before = [d_orig, cc_orig, acc_orig]
        after = [d_pert, cc_pert, acc_dp]

        x = np.arange(len(labels_bar))
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width / 2, before, width, label='Before DP', color='skyblue')
        ax.bar(x + width / 2, after, width, label='After DP', color='salmon')

        ax.set_ylabel('Values')
        ax.set_title('Comparison: Before vs After Differential Privacy')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar)
        ax.legend()

        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    threading.Thread(target=task).start()

# GUI Buttons
Button(main, text="Upload Dataset", command=uploadDataset, bg='lightblue', font=('Arial', 12)).place(x=50, y=50)
Button(main, text="Generate Perturbed Data", command=purturbedData, bg='orange', font=('Arial', 12)).place(x=220, y=50)
Button(main, text="Compare Graphs", command=compareGraphs, bg='green', font=('Arial', 12)).place(x=420, y=50)

main.mainloop()
