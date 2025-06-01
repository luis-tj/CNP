from CNP_GA import *
import ast
import multiprocessing
import sys
import os
import pickle
import time
import collections
import json
import copy



class Solver_CNP:
    def __init__(self, _G, _K, _nIters, _pop, _pC, _pM, _el, _pop_num, _alpha, _ret_rate, _gene_ops, _solv_method, _prun_method, _rand_inds, _combine, _features_arrays, _avgByIt_Run, _greedy_feat):
        self.G                 = _G
        self.num_nodes         = self.G.number_of_nodes()
        self.node_pool         = list(self.G.nodes())
        self.node_pool_best    = []
        self.K                 = _K
        
        self.nIters            = int(_nIters)
        self.popSize           = int(_pop)
        self.pC                = float(_pC)
        self.pM                = float(_pM)
        self.el                = float(_el)
        
        self.pop_num           = int(_pop_num)
        self.alpha             = float(_alpha)
        self.ret_rate          = float(_ret_rate)
        self.gene_ops          = int(_gene_ops)
        self.solv_method       = _solv_method
        self.prun_method       = _prun_method
        self.rand_inds         = round(float(_rand_inds) * self.popSize)
        self.combine           = int(_combine)        
        self.features          = _features_arrays
        self.avgByIt_Run       = int(_avgByIt_Run)
        self.pruning_time      = 'noPrune'
        
        self.n_features        = len(self.features)
        self.n_to_prune        = int(self.alpha * self.num_nodes)
        self.pruning_iters     = round((1 - self.ret_rate) / self.alpha) if self.prun_method == 'worst' else round(self.ret_rate / self.alpha)
        self.greedy_feat       = _greedy_feat
        self.greedy_feat_rev   = self.greedy_feat[::-1] if self.greedy_feat else []
    
        self.num_processors    = 4

    def search(self):
        if self.solv_method == 'noPrune':
            return self.noPruning()
        
        elif self.solv_method == 'popIn':
            return self.popInitial()
        
        elif self.solv_method == 'popEv':
            return self.popEvolution()
                
        elif self.solv_method == 'Research':
            return self.researchMethod()
        

    def noPruning(self):
        print('No pruning:')
        # Main GA
        GA = GA_CNP(self.G, self.node_pool, self.K, self.nIters, self.popSize, self.pC, self.pM, self.el, self.avgByIt_Run, 1, [])
        bestObj, objInit, bestSol, listBestByIt, listAvgByIt = GA.search()
        
        return bestObj, objInit, bestSol, listBestByIt, listAvgByIt, self.pruning_time


    def popInitial(self):
        print('Pop Initial:')
        # Pruning stage
        t0_prune = time.perf_counter()
        for _ in range(self.pruning_iters):
            populations = [GA_CNP(self.G, self.node_pool, self.K, self.gene_ops, self.popSize, self.pC, self.pM, self.el, 0, 0, []).population for _ in range(self.pop_num)]
            
            # Getting best inds from pops in parallel
            with multiprocessing.Pool(processes=self.num_processors) as pool:
                best_inds = pool.map(self.minInPop, populations)
            
            # Voting and pruning
            votes_nodes = collections.Counter(node for list_pop_i in best_inds for node in list_pop_i)
            self.pruneNodePool(votes_nodes)
            print('Pruning done. Size node pool:',int(round(100 * len(self.node_pool) / self.num_nodes, 0)),'%')
        t1_prune = time.perf_counter()

        # Recording pruning time
        self.pruning_time = t1_prune - t0_prune

        # Main GA
        print('Entering main GA:')
        GA = GA_CNP(self.G, self.node_pool, self.K, self.nIters, self.popSize, self.pC, self.pM, self.el, self.avgByIt_Run, 1, [])
        bestObj, objInit, bestSol, listBestByIt, listAvgByIt = GA.search()
        
        return bestObj, objInit, bestSol, listBestByIt, listAvgByIt, self.pruning_time


    def minInPop(self, pop):
        return min(pop, key=lambda ind: ind.getFitness()).genes


    def popEvolution(self):
        print('Pop Evolution:')
        # Pruning stage
        t0_prune = time.perf_counter()
        for _ in range(self.pruning_iters):
            GAs = [GA_CNP(self.G, self.node_pool, self.K, self.gene_ops, self.popSize, self.pC, self.pM, self.el, 0, 0, []) for _ in range(self.pop_num)]
            
            # Getting best inds from pops in parallel
            with multiprocessing.Pool(processes=self.num_processors) as pool:
                best_inds = pool.map(self.evolvePop, GAs)
            
            # Voting and pruning
            votes_nodes = collections.Counter(node for list_pop_i in best_inds for node in list_pop_i)
            self.pruneNodePool(votes_nodes)
            print('Pruning done. Size node pool:',int(round(100 * len(self.node_pool) / self.num_nodes, 0)),'%')
        t1_prune = time.perf_counter()

        # Recording pruning time
        self.pruning_time = t1_prune - t0_prune

        # Main GA
        print('Entering main GA:')
        GA = GA_CNP(self.G, self.node_pool, self.K, self.nIters, self.popSize, self.pC, self.pM, self.el, self.avgByIt_Run, 1, [])
        bestObj, objInit, bestSol, listBestByIt, listAvgByIt = GA.search()
        
        return bestObj, objInit, bestSol, listBestByIt, listAvgByIt, self.pruning_time


    def evolvePop(self, GA):
        # Genetic operations for popEv and Research
        _, _, sol, _, _ = GA.search()
        return sol


    def researchMethod(self):
        print(f'Research: combine="{self.combine}", prune_method="{self.prun_method}"')
        
        inds_features = self.popSize - self.rand_inds                                        # Inds per feature if not combined to each other in the initialization (one pop per feature)
        inds_feature_i = [inds_features // self.n_features for _ in range(self.n_features)]  # Inds per feature if combined to each other (pop comes from all features)

        # Making sure sum(inds_feature_i) + self.rand_inds = self.popSize (by adding inds into the features in order)
        i = 0
        while True:
            if sum(inds_feature_i) + self.rand_inds < self.popSize:
                inds_feature_i[i] += 1
                i = (i + 1) if (i + 1) < self.n_features else 0
            else:
                break

        # Pruning stage
        t0_prune = time.perf_counter()
        for _ in range(self.pruning_iters):
            
            # To store chromosomes for each population's initialization
            pops = [[] for _ in range(self.pop_num)]

            # Features are combined in each population's initialization
            if self.combine:
                
                for p in range(self.pop_num):
                    for i in range(self.n_features):
                        surviving_idxs = np.where(np.isin(self.features[i][0], np.array(self.node_pool)))[0]

                        self.features[i][0] = self.features[i][0][surviving_idxs] # nodes
                        self.features[i][1] = self.features[i][1][surviving_idxs] # scores

                        # Add an epsilon to avoid 0 probabilities
                        adj_vals = self.features[i][1] + 0.00000001

                        # Calculate probabilities
                        probs_feature_i = adj_vals / adj_vals.sum()

                        # Sample inds for pop p from feature i
                        pops[p].extend([list(np.random.choice(self.features[i][0], size=self.K, replace=False, p=probs_feature_i)) for _ in range(inds_feature_i[i])])

                    # Adding unformly randomed inds
                    if self.rand_inds > 0:
                        pops[p].extend([list(np.random.choice(self.node_pool, size=self.K, replace=False)) for _ in range(self.rand_inds)])

            # One feature per population
            else:
                for p in range(self.pop_num):
                    surviving_idxs = np.where(np.isin(self.features[p][0], np.array(self.node_pool)))[0]

                    self.features[p][0] = self.features[p][0][surviving_idxs] # nodes
                    self.features[p][1] = self.features[p][1][surviving_idxs] # scores

                    # Add an epsilon to avoid 0 probabilities
                    adj_vals = self.features[p][1] + 0.00000001

                    # Calculate probabilities
                    probs_feature_i = adj_vals / adj_vals.sum()

                    # Sample inds for pop p
                    pops[p].extend([list(np.random.choice(self.features[p][0], size=self.K, replace=False, p=probs_feature_i)) for _ in range(inds_features)])

                    # Adding unformly randomed inds
                    if self.rand_inds > 0:
                        pops[p].extend([list(np.random.choice(self.node_pool, size=self.K, replace=False)) for _ in range(self.rand_inds)])

            # Evolve each population
            GAs = [GA_CNP(self.G, self.node_pool, self.K, self.gene_ops, self.popSize, self.pC, self.pM, self.el, 0, 0, pops[p]) for p in range(self.pop_num)]

            # Getting best inds from pops in parallel
            with multiprocessing.Pool(processes=self.num_processors) as pool:
                best_inds = pool.map(self.evolvePop, GAs)

            # Voting and pruning
            votes_nodes = collections.Counter(node for list_pop_i in best_inds for node in list_pop_i)
            self.pruneNodePool(votes_nodes)
            print('Pruning done. Size node pool:',int(round(100 * len(self.node_pool) / self.num_nodes, 0)),'%')
        t1_prune = time.perf_counter()

        # Recording pruning time
        self.pruning_time = t1_prune - t0_prune

        # Main GA
        print('Entering main GA:')
        if self.prun_method == 'best': # Main GA will work over node_pool_best if prun_method equals to 'best'
            self.node_pool = self.node_pool_best

        GA = GA_CNP(self.G, self.node_pool, self.K, self.nIters, self.popSize, self.pC, self.pM, self.el, self.avgByIt_Run, 1, [])
        bestObj, objInit, bestSol, listBestByIt, listAvgByIt = GA.search()
        
        return bestObj, objInit, bestSol, listBestByIt, listAvgByIt, self.pruning_time


    def pruneNodePool(self, votes_nodes):
        # By default, the least voted nodes from the node pool are removed on each iteration of the pruning stage (worst).
        # solv_method must be 'Research' and prun_method must be 'best' in order to keep the highest voted nodes for the main GA to be solved afterwards (best)
        scores_by_node = {node: votes_nodes[node] for node in self.node_pool}
        
        nodes_by_score = {score: [] for score in scores_by_node.values()}
        for node, score in scores_by_node.items():
            nodes_by_score[score].append(node)

        # If prun_method is 'best', then nodes_by_score is sorted in reverse so that the pruning is applied to the best nodes which can be stored later
        if self.solv_method == 'Research' and self.prun_method == 'best':
            nodes_by_score = sorted(nodes_by_score.items(), reverse=True)
        else:
            nodes_by_score = sorted(nodes_by_score.items())

        # Obtaining the nodes to remove from the node pool
        nodes_to_remove = []

        for score, node_list in nodes_by_score:
            if len(node_list) <= self.n_to_prune - len(nodes_to_remove):
                nodes_to_remove.extend(node_list)
                if len(nodes_to_remove) == self.n_to_prune:
                    break
            else:
                if self.greedy_feat and self.solv_method == 'Research' and self.prun_method == 'best': # Greedy-based tiebreaking (best)
                    nodes_to_remove.extend([node for node, feat_value in self.greedy_feat if node not in nodes_to_remove and node in node_list][:self.n_to_prune - len(nodes_to_remove)])
                elif self.greedy_feat and self.solv_method == 'Research' and self.prun_method == 'worst': # Greedy-based tiebreaking (worst)
                    nodes_to_remove.extend([node for node, feat_value in self.greedy_feat_rev if node not in nodes_to_remove and node in node_list][:self.n_to_prune - len(nodes_to_remove)])
                else:
                    nodes_to_remove.extend(random.sample(node_list, self.n_to_prune - len(nodes_to_remove))) # Random tiebreaking
                break

        # This is the node pool passed to the next iteration of the pruning stage. It will also be the one to use in the main GA if prun_method is 'worst' (provided solv_method is 'Research')
        nodes_to_remove_set = set(nodes_to_remove)
        self.node_pool = [node for node in self.node_pool if node not in nodes_to_remove_set]

        # For the greedy strategy of popIn and popEv in the code provided by Yu et al.
        if self.greedy_feat and self.solv_method in ['popIn', 'popEv']:
            lenght_node_pool = len(self.node_pool)
            self.node_pool = self.node_pool[:round(lenght_node_pool/2)]
            self.node_pool = (self.node_pool + [node for node, feat_value in self.greedy_feat if node not in self.node_pool])[:lenght_node_pool]

        # node_pool_best is the node pool used in the main GA after the pruning stage (only if solv_method is 'Research' and prun_method is 'best')
        if self.solv_method == 'Research' and self.prun_method == 'best':
            self.node_pool_best.extend(nodes_to_remove)



def my_main():
    """
    CNP_Solver.py args:
        G_file          : e.g. ErdosRenyi_n500
        nRuns           : number of runs to perform. Default 10
        nIters          : number of generations to be produced in a run of the main GA. Default 5000
        pop             : number of chromosomes in the population. Default 100
        pC              : crossover probability (between 0 and 1). Default 0.2
        pM              : mutation probability (between 0 and 1). Default 0.6
        el              : proportion of population that are elite (between 0 and 1). Default 0.1
        pop_num         : number of populations. Default 10
        alpha           : deletion rate of nodes from the node pool (between 0 and 1). Default 0.2
        ret_rate        : retention rate of nodes from the original node pool (from the full set of nodes). Default 0.4
        gene_ops        : number of genetic iterations for the popEvolution method. Default 30
        solving_method  : noPrune, popIn, popEv, Research
        pruning_method  : worst, best
        random_inds     : percentage of random chromosomes in each population (between 0 and 1)
        combine_feats   : wether to combine features to initialize populations (1) or not (0)
        greedy_feat_idx : index of the feature selected to greedily complete the half of the node pool (Yu's popIn, popEv) or to complete the nodes in each pruning iterations in Research, combining the nodes produced from the voting with the best ones according to this feature. e.g. 4 for degree_centrality, N for no greedy_feat
        features        : list of feautres to form the populations from. e.g. [0,4,6,9] (no spaces between brackets, commas and numbers). Number must be indexes from the list of features below

    * if combine_feats == no_comb, then len(features) must be equal to pop_num

    Features:
        'betweenness_centrality',       (0)
        'closeness_centrality',         (1)
        'clustered_local_degree',       (2)
        'clustering_coefficient',       (-) Do not use this one (inconsistent feature)
        'degree_centrality',            (4)
        'eigenvector_centrality',       (5)
        'harmonic_centrality',          (6)
        'k_shell',                      (7)
        'leverage_centrality',          (8)
        'load_centrality',              (9)
        'local_centrality',             (10)
        'pagerank',                     (11)
        'quasi_laplacian_centrality',   (12)
        'spon_centrality',              (13)
        'subgraph_centrality',          (14)
        'third_laplacian_energy'        (15)

    Tests
        e.g.:
        python CNP_Solver.py ErdosRenyi_n500 10 5000 100 0.2 0.6 0.1 10 0.2 0.4 30 Research worst 0.2 1 N [0,4,6,9]
    """

    # Retrieve parameters from command line
    G_file, nRuns, nIters, pop, pC, pM, el, pop_num, alpha, ret_rate, gene_ops, solv_method, prun_method, rand_inds, combine, greedy_feat_idx, feats = sys.argv[1:]

    # Load desired Graph and number of Critical Nodes
    cwd = os.getcwd()
    folder_graphs = os.path.join(cwd, 'Graphs pickle Features')
    G_file = os.path.join(folder_graphs, G_file + '.pickle')

    # Graph
    with open(G_file, 'rb') as file:
        G = pickle.load(file)

    # Features
    features_idxs = ast.literal_eval(feats)

    # List of available features
    list_of_features = ['betweenness_centrality',       # 0
                        'closeness_centrality',         # 1
                        'clustered_local_degree',       # 2
                        'clustering_coefficient',       # - Do not use this one (it is not consistent)
                        'degree_centrality',            # 4
                        'eigenvector_centrality',       # 5
                        'harmonic_centrality',          # 6
                        'k_shell',                      # 7
                        'leverage_centrality',          # 8
                        'load_centrality',              # 9
                        'local_centrality',             # 10
                        'pagerank',                     # 11
                        'quasi_laplacian_centrality',   # 12
                        'spon_centrality',              # 13
                        'subgraph_centrality',          # 14
                        'third_laplacian_energy']       # 15
    
    features = [list_of_features[i] for i in features_idxs]
    
    # Exit if pop_num != len(features) when not combining features in Research mode
    if solv_method == 'Research' and not int(combine) and int(pop_num) != len(features):
        sys.exit('Error: pop_num != len(features)')
    
    # Normalize features into the [0, 1] range for safely adding epsilon later on in the sampling process. If feature is uniform in values, then skip normalization
    for feat in features:
        min_value = min(G['Features'][feat][0].values())
        max_value = max(G['Features'][feat][0].values())
        if max_value != min_value:
            G['Features'][feat][0] = {node: (value - min_value) / (max_value - min_value) for node, value in G['Features'][feat][0].items()}

    # Creating array for efficient probaibility computation for sampling populations in Research mode
    features_dicts = [dict(sorted(G['Features'][feat][0].items())) for feat in features]
    features_arrays = [[np.array(list(d.keys())), np.array(list(d.values()))] for d in features_dicts]

    # For the greedy strategy of popIn and popEv in the code provided by Yu et al. (half genes from voting + half genes from best degree) or for Research 
    greedy_feat_name = list_of_features[int(greedy_feat_idx)] if greedy_feat_idx != 'N' else 'No'
    if greedy_feat_name != 'No':
        greedy_feat = sorted(G['Features'][greedy_feat_name][0].items(), key=lambda x: x[1], reverse=True)
    else:
        greedy_feat = []

    # To store the results
    results_dict = {'Param': {'G': os.path.basename(G_file),
                              'K': G['K'],
                              'nRuns': int(nRuns),
                              'nIters': int(nIters),
                              'popSize': int(pop),
                              'pC': float(pC),
                              'pM': float(pM),
                              'el': float(el),
                              'pop_num': int(pop_num),
                              'alpha': float(alpha),
                              'ret_rate': float(ret_rate),
                              'gene_ops': int(gene_ops),
                              'solv_method': solv_method,
                              'prun_method': prun_method,
                              'rand_inds': float(rand_inds),
                              'combine': int(combine),
                              'features': features,
                              'greedy_feat': greedy_feat_name},
                    'RunsResults': {i + 1: {'BestPCList': None,
                                            'AvgPCList': None,
                                            'BestPC': None,
                                            'Runtime': None,
                                            'PruningTime': None} for i in range(int(nRuns))},
                    'GAResults': {'BestSol': None,
                                  'BestPC': None,
                                  'AvgPC': None,
                                  'AvgInitPC': None,
                                  'AvgRuntime': None,}}

    # Set seed for reproductability
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Run the algorithm
    avgTime = 0
    nRuns = int(nRuns)

    # Two random runs for which avgByIt will be calculated
    runs_avgByIt = []#random.sample(range(1, nRuns + 1), 2) # Uncomment this for last experiments

    # 1st run
    print("Run 1")
    avgByIt_Run = 1 if 1 in runs_avgByIt else 0
    t0 = time.perf_counter()
    CNP = Solver_CNP(G['Graph'], G['K'], nIters, pop, pC, pM, el, pop_num, alpha, ret_rate, gene_ops, solv_method, prun_method, rand_inds, combine, copy.deepcopy(features_arrays), avgByIt_Run, greedy_feat)
    bestObj, objInit, bestSol, listBestByIt, listAvgByIt, pruningTime = CNP.search()
    t1 = time.perf_counter()
    print("Cost Run 1: ",bestObj)

    results_dict['RunsResults'][1]['BestPCList'] = listBestByIt
    results_dict['RunsResults'][1]['AvgPCList'] = listAvgByIt
    results_dict['RunsResults'][1]['BestPC'] = int(bestObj)
    results_dict['RunsResults'][1]['Runtime'] = t1 - t0
    results_dict['RunsResults'][1]['PruningTime'] = pruningTime

    # Check if solution found on 1st run is valid
    avgTime += t1 - t0
    avgObj, avgInitObj = bestObj, objInit
    for i in range(1, nRuns):
        random.seed(seed + i * 100)
        np.random.seed(seed + i * 100)
        
        # i-th run
        print("Run ",i + 1)
        avgByIt_Run = 1 if (i + 1) in runs_avgByIt else 0
        t0 = time.perf_counter()
        CNP = Solver_CNP(G['Graph'], G['K'], nIters, pop, pC, pM, el, pop_num, alpha, ret_rate, gene_ops, solv_method, prun_method, rand_inds, combine, copy.deepcopy(features_arrays), avgByIt_Run, greedy_feat)
        obj, objInit, sol, listBestByIt, listAvgByIt, pruningTime = CNP.search()
        t1 = time.perf_counter()
        print("Cost Run ",i + 1, ": ",obj)

        results_dict['RunsResults'][i + 1]['BestPCList'] = listBestByIt
        results_dict['RunsResults'][i + 1]['AvgPCList'] = listAvgByIt
        results_dict['RunsResults'][i + 1]['BestPC'] = int(obj)
        results_dict['RunsResults'][i + 1]['Runtime'] = t1 - t0
        results_dict['RunsResults'][i + 1]['PruningTime'] = pruningTime

        # Check if the solution found on the i-th run is valid
        avgTime += t1 - t0
        avgObj += obj
        avgInitObj += objInit
        if obj < bestObj:
            bestObj = obj
            bestSol = sol

    print("Best Sol: ",bestSol)
    print("Cost Best Sol: ",bestObj)

    results_dict['GAResults']['BestSol'] = bestSol
    results_dict['GAResults']['BestPC'] = int(bestObj)
    results_dict['GAResults']['AvgPC'] = round(avgObj / nRuns, 2)
    results_dict['GAResults']['AvgInitPC'] = round(avgInitObj / nRuns, 2)
    results_dict['GAResults']['AvgRuntime'] = round(avgTime / nRuns, 2)

    # Store results
    file_name = f'{os.path.basename(G_file)[:-7]}_{G['K']}_{nRuns}_{nIters}_{pop}_{pC}_{pM}_{el}_{pop_num}_{alpha}_{ret_rate}_{gene_ops}_{solv_method}_{prun_method}_{rand_inds}_{combine}_{greedy_feat_idx}_{feats}'
    folder_results = os.path.join(os.getcwd(), 'Results')
    
    # Pickle
    with open(os.path.join(folder_results, file_name + '.pickle'), 'wb') as file:
        pickle.dump(results_dict, file)

    # Json
    with open(os.path.join(folder_results, file_name + '.json'), 'w') as file:
        json.dump(results_dict, file)



if __name__ == '__main__':
    my_main()
