from CNP_Individual import *
import numpy as np
import heapq



class GA_CNP:
    def __init__(self, _G, _node_pool, _K, _maxIterations, _popSize, _xoverProb, _mutationRate, _elites, _avgByIt, _display, _genes_pop=[]):
        
        self.G                   = _G
        self.node_pool           = _node_pool
        self.genSize             = _K
        
        self.maxIterations       = _maxIterations
        self.popSize             = _popSize
        self.crossoverProb       = _xoverProb
        self.mutationRate        = _mutationRate
        self.elites              = round(self.popSize * _elites)
        self.addAvgByIt          = int(_avgByIt)
        self.display             = int(_display)
        self.genes_pop          = _genes_pop

        self.iteration           = 0
        self.population          = []
        self.best                = None
        self.bestInitSol         = self.initPopulation()
        
        self.bestByIt            = [(0, self.bestInitSol)]
        self.avgByIt             = [(0, np.mean([ind.getFitness() for ind in self.population]))] if self.addAvgByIt else []



    def initPopulation(self):
        # This is for random initialization (only for noPrune, popIn and popEv)
        if not self.genes_pop:
            self.genes_pop = [[] for _ in range(self.popSize)]

        # Initialize populations wether it is randomly or product of a pruning strategy (genes coming from worst or best strategies)
        best_fitness = np.inf
        for i in range(0, self.popSize):
            individual = Individual(self.G, self.node_pool, self.genSize, self.genes_pop[i])
            individual.computeFitness()
            self.population.append(individual)

            if individual.getFitness() < best_fitness:
                best_fitness = individual.getFitness()
                best_idx = i

        self.best = self.population[best_idx].copy()
        return self.best.getFitness()


    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            if self.display:
                print('Improve on it: ',self.iteration + 1,' from ',self.best.getFitness(),' to ',candidate.getFitness())
            self.best = candidate.copy()


    def binaryTournamentSelection(self):
        # Total chrmosomes (parents) to be selected. Each pair of parents will generate 2 children
        extra_inds = self.popSize - self.elites
        if extra_inds % 2 != 0:
            extra_inds += 1

        # Number of candidates across all tournaments (binary)
        n_candidates = extra_inds * 2

        # Candidates selected with replacement
        candidates_idxs = list(np.random.randint(0, self.popSize, size=n_candidates))

        # Tournaments
        tournaments = [(candidates_idxs[i], candidates_idxs[i+1]) for i in range(0, len(candidates_idxs), 2)]

        # Get surviving indexes from the tournaments
        return [self.population[idx1].copy() if self.population[idx1].getFitness() <= self.population[idx2].getFitness() else self.population[idx2].copy() for idx1, idx2 in tournaments]

        
    def crossover(self, indA, indB):
        # 2-Children xover
        # Uniform crossover
        if random.random() > self.crossoverProb:
            child1 = Individual(self.G, self.node_pool, self.genSize, indA)
            child2 = Individual(self.G, self.node_pool, self.genSize, indB)
            return child1, child2
        
        swaps_probs = np.random.rand(self.genSize)
        swaps_idxs = np.where(swaps_probs <= 0.5)[0]

        genes1 = list(set([indB[i] if i in swaps_idxs else indA[i] for i in range(self.genSize)]))
        genes2 = list(set([indA[i] if i in swaps_idxs else indB[i] for i in range(self.genSize)]))

        # Preserve uniqueness
        if len(genes1) < self.genSize:
            genes1.extend([i for i in indB if i not in genes1])
        if len(genes2) < self.genSize:
            genes2.extend([i for i in indA if i not in genes2])

        child1 = Individual(self.G, self.node_pool, self.genSize, genes1)
        child2 = Individual(self.G, self.node_pool, self.genSize, genes2)
        return child1, child2
    

    def mutation(self, ind):
        for index in range(self.genSize):
            if random.random() > self.mutationRate:
                continue
                
            # Mutate gene, ensuring uniqueness in the chromosome
            while True:
                new_gene = random.choice(self.node_pool)
                if new_gene not in ind.genes:
                    ind.genes[index] = new_gene
                    break


    def newGeneration(self):
        # Chromosomes from selection
        selected_inds = self.binaryTournamentSelection()
        
        # Form pairs of parents
        pairs_of_parents = [(selected_inds[i], selected_inds[i+1]) for i in range(0, len(selected_inds), 2)]

        # Genetic operations
        i = self.elites
        for ind1, ind2 in pairs_of_parents[:-1]: # All pairs but last one
            child1, child2 = self.crossover(ind1.genes, ind2.genes)
            for child in [child1, child2]:
                self.mutation(child)
                child.computeFitness()
                self.updateBest(child)
                self.population[i] = child
                i += 1

        ind1, ind2 = pairs_of_parents[-1] # Last pair. Add child2 only if there is still room in the population
        child1, child2 = self.crossover(ind1.genes, ind2.genes)
        for child in [child1, child2]:
            self.mutation(child)
            child.computeFitness()
            self.updateBest(child)
            self.population[i] = child
            if (self.popSize - self.elites) % 2 != 0:
                break
            i += 1


    def GAStep(self):
        # Promote elite chromosomes to next generation and compute the rest with self.newGeneration()
        elites_to_next_gen = heapq.nsmallest(self.elites, self.population, key=lambda ind: ind.getFitness())
        self.newGeneration()
        self.population[:self.elites] = elites_to_next_gen


    def search(self):
        self.iteration = 0
        while self.best.getFitness() > 0 and self.iteration < self.maxIterations:
            self.GAStep()
            self.bestByIt.append((self.iteration + 1, self.best.getFitness()))
            if self.addAvgByIt:
                self.avgByIt.append((self.iteration + 1, np.mean([ind.getFitness() for ind in self.population])))
            self.iteration += 1

        return self.best.getFitness(), self.bestInitSol, self.best.genes, self.bestByIt, self.avgByIt
