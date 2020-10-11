import numpy as np


class KontrollQuerschnitt:
    def __init__(self):
        self.name = None
        self.p0 = None  # Start Point
        self.p1 = None  # End Point
        self.elmt_ids = None  # Elements, where p0 and p1 are within
        self.edge_ids = None  # Edges that are intersected
        self.intersections = None

    def to_numpy(self):
        return np.array([self.p0, self.p1])

    # At the moment, only the start and end point are imported
    def from_lineshape(self, name, geometry):
        self.name = name
        self.p0 = (geometry.points[0][0], geometry.points[0][1])
        self.p1 = (geometry.points[-1][0], geometry.points[-1][1])

        if len(geometry.points) > 2:
            print("WARNUNG: Nur der Anfangs- und End-Punkt des "
                  "Kontrollquerschnitts wird gelesen.")


class Mesh:
    def __init__(self, nodes_array=None, elements=None, edges=None):
        self.nodes_array = nodes_array
        self.elements = elements
        self.edges = edges
        self.node_elmt_link = None
        self.node_edge_link = None
