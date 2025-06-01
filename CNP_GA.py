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


    def rouletteSelection(self):
        # Get the fitness of every chromosome
        fitness_inds = np.array([[ind.getFitness() for ind in self.population]])

        # Get the maximum and modify the fitness of each chromosome. Modification responds to minimization problem
        max_fitness = np.max(fitness_inds)                                        
        modified_fitness_inds = max_fitness + 1 - fitness_inds

        # Calculate probabilities for the Roulette
        prob_inds = modified_fitness_inds / np.sum(modified_fitness_inds)
        cummulative_prob_inds = np.cumsum(prob_inds)

        # Total chrmosomes (parents) to be selected. Each pair of parents will generate 2 children
        extra_inds = self.popSize - self.elites
        if extra_inds % 2 != 0:
            extra_inds += 1

        # Spin Roulette "extra_inds" times
        spins = np.random.rand(extra_inds)
        inds_idxs = np.searchsorted(cummulative_prob_inds, spins)
        
        return [self.population[i].copy() for i in inds_idxs]

        
    def crossover(self, indA, indB):
        # 2-Children xover
        if random.random() > self.crossoverProb:
            child1 = Individual(self.G, self.node_pool, self.genSize, indA)
            child2 = Individual(self.G, self.node_pool, self.genSize, indB)
            return child1, child2
        
        midP = random.randint(1, self.genSize - 2)
        p1 = indA[0:midP]
        p2 = indB[0:midP]

        genes1 = (p1 + [i for i in indB if i not in p1])[:self.genSize]
        genes2 = (p2 + [i for i in indA if i not in p2])[:self.genSize]
        
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
        # Chromosomes from Roulette selection
        selected_inds = self.rouletteSelection()
        
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
        self.population[:self.elites] = heapq.nsmallest(self.elites, self.population, key=lambda ind: ind.getFitness())
        self.newGeneration()


    def search(self):
        self.iteration = 0
        while self.best.getFitness() > 0 and self.iteration < self.maxIterations:
            self.GAStep()
            self.bestByIt.append((self.iteration + 1, self.best.getFitness()))
            if self.addAvgByIt:
                self.avgByIt.append((self.iteration + 1, np.mean([ind.getFitness() for ind in self.population])))
            self.iteration += 1

        return self.best.getFitness(), self.bestInitSol, self.best.genes, self.bestByIt, self.avgByIt
