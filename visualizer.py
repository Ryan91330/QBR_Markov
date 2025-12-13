import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from subprocess import check_call
import matplotlib.image as mpimg


class MarkovPlotter:
    def __init__(self, P, labels):
        self.P = np.asarray(P)
        self.labels = labels
        self.n = len(labels)

    def draw(self):
        # Build MultiDiGraph
        G = nx.MultiDiGraph()
        edge_labels = {}

        # Use labels directly instead of tuple nodes
        states = self.labels

        # Add edges exactly like before
        for i, origin in enumerate(states):
            for j, dest in enumerate(states):
                prob = self.P[i][j]

                if prob > 0:
                    G.add_edge(
                        origin,
                        dest,
                        weight=prob,
                        label=f"{round(prob,3)}"
                    )
                    edge_labels[(origin, dest)] = f"{prob:.02f}"

        # --- DOT export
        write_dot(G, 'mc.dot')

        # Run graphviz to render png
        output_png = 'mc_plot.png'
        check_call(['dot', '-Tpng', 'mc.dot', '-o', output_png])

        # Load and display PNG
        img = mpimg.imread(output_png)
        plt.figure(figsize=(10, 7))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
