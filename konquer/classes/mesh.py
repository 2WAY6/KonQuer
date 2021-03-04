import sys
import time

import numpy as np


class Mesh:
    def __init__(self, nodes_array=None, elements=None, edges=None):
        self.nodes_array = nodes_array
        self.elements = elements
        self.edges = edges
        self.node_elmt_link = None
        self.node_edge_link = None

    def import_mesh(self, path_mesh):
        t0 = time.time()
        if path_mesh.endswith("2dm"):
            self.import_2dm_mesh(path_mesh)
        elif path_mesh.endswith("dat"):
            self.import_uro_mesh(path_mesh)
        else:
            sys.exit("Netzformat nicht unterstuetzt. Programmabbruch.")
        print("  - {} Knoten, {} Elemente und {} Kanten importiert.".format(
            len(self.nodes_array), len(self.elements), len(self.edges)))
        print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

        print("- Verknuepfe Kanten und Knoten...")
        t0 = time.time()
        node_edge_link = [[] for i in range(self.nodes_array.shape[0])]
        for ei, edge in enumerate(self.edges):
            node_edge_link[edge[0]].append(ei)
            node_edge_link[edge[1]].append(ei)
        print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

        print("- Verknuepfe Elemente und Knoten...")
        t0 = time.time()
        node_elmt_link = [[] for i in range(self.nodes_array.shape[0])]
        for eid, node_ids in enumerate(self.elements):
            for nid in node_ids:
                node_elmt_link[nid].append(eid)
        for nid, eids in enumerate(node_elmt_link):
            node_elmt_link[nid] = list(set(eids))
        print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

        self.node_edge_link = node_edge_link
        self.node_elmt_link = node_elmt_link

    def import_2dm_mesh(self, path_mesh):
        print("- Lese 2dm-Mesh...")
        nodes = []
        edges = []
        elements = []
        for line in open(path_mesh):
            if line.startswith("ND "):
                nodes.append(list(map(float, line.split()[2:])))
            elif line.startswith("E3T") or line.startswith("E4Q"):
                if line.startswith("E3T"):
                    nd_indices = [int(nid) - 1 for nid in line.split()[2:5]]
                else:
                    nd_indices = [int(nid) - 1 for nid in line.split()[2:6]]
                # nd_indices = [int(nid) - 1 for nid in line.split()[2:-1]]
                elements.append(nd_indices)
                c = len(nd_indices)
                for i in range(c):
                    min_i = min(nd_indices[i], nd_indices[(i + 1) % c])
                    max_i = max(nd_indices[i], nd_indices[(i + 1) % c])
                    edges.append(f"{min_i} {max_i}")

        edges = np.array([[int(edge.split()[0]), int(edge.split()[1])] for edge in list(set(edges))])

        nodes_array = np.zeros((len(nodes), 5))
        nodes_array[:, (0, 1, 2)] = np.array(nodes)

        self.nodes_array = nodes_array
        self.edges = edges
        self.elements = elements

    def import_uro_mesh(self, path_mesh):
        print("- Lese UnRunOff-Mesh...")
        lines = [line for line in open(path_mesh).readlines() if not line.startswith('C')]

        n_bndnodes = int(lines[0])
        n_nodes = int(lines[1])
        n_elmts = int(lines[2 + n_nodes + n_bndnodes])

        i_nodes = [2, 2 + n_nodes + n_bndnodes]
        i_elmts = [2 + n_nodes + n_bndnodes + 1, 2 + n_nodes + n_bndnodes + 1 + n_elmts]

        def line_to_node(line):
            return [float(x) for x in line.split()[1:]]

        def line_to_elmt(line):
            return [int(nid) for nid in line.split()[:3]]

        nodes = [line_to_node(line) for line in lines[i_nodes[0]:i_nodes[1]]]
        nodes_array = np.zeros((len(nodes), 5))
        nodes_array[:,(0,1,2)] = np.array(nodes)

        elements = [line_to_elmt(line) for line in lines[i_elmts[0]:i_elmts[1]]]

        edges = []
        for nids in elements:
            for i in range(3):
                min_i = min(nids[i], nids[(i + 1) % 3])
                max_i = max(nids[i], nids[(i + 1) % 3])
                edges.append(f"{min_i} {max_i}")
        edges = np.array([[int(edge.split()[0]), int(edge.split()[1])] for edge in list(set(edges))])

        self.nodes_array = nodes_array
        self.edges = edges
        self.elements = elements
