import networkx as nx
import random



class Individual:
    def __init__(self, _G, _node_pool, _K, genes=[]):

        self.G          = _G
        self.node_pool  = _node_pool
        self.genSize    = _K
        self.genes      = []
        self.fitness    = None

        if genes: # Child genes from crossover
            self.genes = genes

        else:   # Random initialisation of genes
            self.genes = random.sample(self.node_pool, self.genSize)



    def copy(self):
        ind = Individual(self.G, self.node_pool, self.genSize, self.genes[:self.genSize])
        ind.fitness = self.getFitness()
        return ind


    def getFitness(self):
        return self.fitness


    def computeFitness(self):
        H = self.G.subgraph(set(self.G.nodes) - set(self.genes))
        self.fitness = sum(len(c) * (len(c) - 1) // 2 for c in nx.connected_components(H)) # pairwise_connectivity
