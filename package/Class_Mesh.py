import numpy as np
import matplotlib.pyplot as plt


class Mesh():
    def __init__(self, nb_nodes=0, nb_elt=0, dim=0, elementType=None, nodes=None, connectivity_table=None):
        self.nb_nodes = nb_nodes
        self.nb_elt = nb_elt
        self.dim = dim
        self.elementType = elementType
        self.nodes = nodes
        self.connectivity_table = connectivity_table

    def readmesh(self, filename):
        """Read a .msh file and construct the corresponding Mesh object. Be careful: it only works with the Medit (Inria) format."""

        f = open(filename, 'r')

        # Read the headers to get the interesting informations
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        self.nb_nodes = int(f.readline())

        # Read the nodes
        self.nodes = np.zeros((self.nb_nodes, 3))
        line = f.readline()
        for index_line in range(self.nb_nodes):
            self.nodes[index_line, :] = list(map(float, line.split()))[:3]
            line = f.readline()

        # Read the elements
        # remove the first space and the '\n' at the end
        self.elementType = line[1:-1]
        if self.elementType == 'Triangles':
            self.dim = 2
            self.nb_elt = int(f.readline())
            self.connectivity_table = np.zeros((self.nb_elt, 3), dtype=int)
            line = f.readline()
            for index_line in range(self.nb_elt):
                self.connectivity_table[index_line, :] = list(
                    map(int, line.split()))[:3]
                line = f.readline()

        self.connectivity_table -= 1  # Gmsh starts indexing at 1 and not 0

    def plot(self, ax=None, alpha=1):
        assert self.dim == 2

        nb_edges = np.shape(self.connectivity_table)[1]
        if ax == None:
            fig, ax = plt.subplots()

        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], s=2**4, color='k')

        for index_elt in range(self.nb_elt):
            for i in range(nb_edges):
                x = [self.nodes[self.connectivity_table[index_elt, i], 0],
                     self.nodes[self.connectivity_table[index_elt, (i+1) % nb_edges], 0]]
                y = [self.nodes[self.connectivity_table[index_elt, i], 1],
                     self.nodes[self.connectivity_table[index_elt, (i+1) % nb_edges], 1]]
                ax.plot(x, y, '-k', linewidth=1., alpha=alpha)

        return ax
