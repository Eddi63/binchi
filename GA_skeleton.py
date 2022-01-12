# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import numpy as np

def mutation(a):
    """ swap two random indexes """
    index_1 = np.random.randint(0, len(a))
    index_2 = np.random.randint(0, len(a))
    a[index_1], a[index_2] = a[index_2], a[index_1]
    return a


def select_proportional(Genome, fitness, rand_state):
    ''' RWS: Select one individual out of a population Genome with fitness values fitness using proportional selection.'''
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[0]

    return Genome[idx]

def two_point_crossover(a, b, local_state):
    index_1 = local_state.randint(0, len(a))
    index_2 = local_state.randint(0, len(b))
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    child_1 = np.copy(a[index_1:index_2]) #copying section for cross
    child_2 = np.copy(b[index_1:index_2])
    child_1 = np.concatenate([child_1, [i for i in b if i not in child_1]]) #bookkeeping for legal permuation
    child_2 = np.concatenate([child_2, [i for i in a if i not in child_2]])
    return child_1, child_2


def sequentialConstructiveCrossover(parent1, parent2):
    # Literature reference: "Genetic Algorithm for the Traveling Salesman Problem using Sequential Constructive Crossover
    # Operator." (Zakir H. Ahmed)

    # This function performs Sequential Constructive Crossover operation with the given two parents. And produces
    # one child.

    child = []

    # First, the first node of parent1 is taken to child as first node.
    child.append(parent1[0])
    child_indice = 0

    # Finds the next node of the last selected node of child in both parents. Then calculate the distences between the last
    # node of the child and the next nodes in both parents. Keep the nearest node as the next node in child. Iterate this
    # operation until the child is completed.

    while child_indice < len(parent1) - 1:
        # Find the legitimate node for parent1 which comes after childs last node.
        for node_indice in range(len(parent1)):
            if parent1[node_indice] == child[child_indice]:
                parent1_candidate_indice = node_indice + 1
                if parent1_candidate_indice == len(parent1):
                    parent1_candidate_indice = 0
                if parent1[parent1_candidate_indice] in child:

                    # There is no legitimate node in parent1 so the candidate will be selected sequentially from parent1.
                    parent1_nodes = range(1, len(parent1) + 1)
                    for node in parent1_nodes:
                        indice = find_node(parent1, node)
                        if parent1[indice] not in child:
                            parent1_candidate_indice = indice
                            break

        # Find the legitimate node for parent2 which comes after childs last node.
        for node_indice in range(len(parent2)):
            if parent2[node_indice] == child[child_indice]:
                parent2_candidate_indice = node_indice + 1
                if parent2_candidate_indice == len(parent2):
                    parent2_candidate_indice = 0
                if parent2[parent2_candidate_indice] in child:
                    # There is no legitimate node in parent2 so the candidate will be selected sequentially from parent2.
                    parent2_nodes = range(1, len(parent2) + 1)
                    for node in parent2_nodes:
                        indice = find_node(parent2, node)
                        if parent2[indice] not in child:
                            parent2_candidate_indice = indice
                            break

        # Calculate the distances between the last node of child and the legitimate nodes selected in both parent1 and parent2.
        dist1 = child[child_indice].distance(parent1[parent1_candidate_indice])
        dist2 = child[child_indice].distance(parent2[parent2_candidate_indice])

        # If distance between child last node and parent1's candidate is less than child's last node and parent2's candidate
        # then choose parent1's candidate.And choose parent2's candidate in reverse condition.
        if dist1 < dist2:
            child.append(parent1[parent1_candidate_indice])
        else:
            child.append(parent2[parent2_candidate_indice])
        child_indice = child_indice + 1

    return child

def max_edge_crossover(graph, parent1, parent2, local_state):
    max_dis1, min_dis1 = 0, 10**4
    max_dis2, min_dis2 = 0, 10**4
    child_1 = np.copy(parent1)
    child_2 = np.copy(parent2)
    """ step 1 : find max an min  edge in each parent """
    for vertex in range(len(parent1)):
        if graph[parent1[vertex], parent1[np.mod(vertex + 1, len(parent1))]] > max_dis1:
            v11, v12 = vertex, np.mod(vertex + 1, len(parent1))
            max_dis1 = graph[parent1[vertex], parent1[np.mod(vertex + 1, len(parent1))]]
        if graph[parent1[vertex], parent1[np.mod(vertex + 1, len(parent1))]] < min_dis1:
            m11, m12 = vertex, np.mod(vertex + 1, len(parent1))
            min_dis1 = graph[parent1[vertex], parent1[np.mod(vertex + 1, len(parent1))]]
        if graph[parent2[vertex], parent2[np.mod(vertex + 1, len(parent2))]] > max_dis2:
            v21, v22 = vertex, np.mod(vertex + 1, len(parent2))
            max_dis2 = graph[parent2[vertex], parent2[np.mod(vertex + 1, len(parent2))]]
        if graph[parent2[vertex], parent2[np.mod(vertex + 1, len(parent2))]] < min_dis2:
            m21, m22 = vertex, np.mod(vertex + 1, len(parent2))
            min = graph[parent2[vertex], parent2[np.mod(vertex + 1, len(parent2))]]
    """ step 2: find minimum combination of max edges """
    min_dis = 10 ** 6
    if v11 != v21 and v11 != v22 and v12 != v21 and v12 != v22:
        for i in [v11, v12]:
            for j in [v21, v22]:
                if graph[parent1[i], parent2[j]] < min_dis:
                    min_dis = graph[parent1[i], parent2[j]]
                    index1, index2 = i, j
        child_1[np.where(parent1 == parent2[index2])], child_1[np.mod(index1+1, len(child_1))] = \
            child_1[np.mod(index1+1, len(child_1))], child_1[np.where(parent1 == parent2[index2])]
        child_2[np.where(parent2 == parent1[index1])], child_2[np.mod(index2+1, len(child_2))] = \
            child_2[np.mod(index2+1, len(child_2))], child_2[np.where(parent2 == parent1[index1])]
    if m11 != m21 and m11 != m22 and m12 != m21 and m12 != m22:
        """ insert minimum Edge to both children """
        x, y = np.where(child_1 == m21)[0][0], np.where(child_1 == m22)[0][0]
        child_1[np.mod(x+1, len(child_1))], child_1[y] = child_1[y], child_1[np.mod(x+1, len(child_1))]
        x, y = np.where(child_2 == m11)[0][0], np.where(child_2 == m12)[0][0]
        child_2[np.mod(x+1, len(child_2))], child_2[y] = child_2[y], child_2[np.mod(x+1, len(child_2))]
    if sum(child_1 == parent1) == len(parent1):
        child_1 = local_state.permutation(len(parent1))
    if sum(child_2 == parent2) == len(parent2):
        child_2 = local_state.permutation((len(parent2)))
    return child_1, child_2

def GA(n, max_evals, fitnessfct, graph, seed=None,  selectfct=select_proportional) :
    eval_cntr = 0
    history = []
    fmin = np.inf
    xmin = np.array([n,1])
    #GA params
    mu = 100
    pc = 0.70
    pm = 2 / n
    #    kXO = 1 # 1-point Xover
    local_state = np.random.RandomState(seed)

    
    Genome = np.array([local_state.permutation(n) for _ in range(mu)])

    #choice of alphabet : ints 1-150
    fitnessPop = []

    for i in range(mu):
        fitnessPop.append(fitnessfct(Genome[i], graph))

    eval_cntr += mu
    fcurr_best = fmin = np.min(fitnessPop)
    xmin = Genome[np.argmin(fitnessPop)]
    history.append(fmin)

    while (eval_cntr < max_evals) :
        newGenome = np.empty([mu, n], dtype=int)
        for i in range(int(mu/2)) :
            p1 = selectfct(Genome,fitnessPop,local_state)
            p2 = selectfct(Genome,fitnessPop,local_state)

            if local_state.uniform() < pc :
                #c1, c2 = two_point_crossover(p1,p2, local_state) #crossover
                c1, c2 = two_point_crossover(p1, p2, local_state)  # crossover
            else :
                c1, c2 = np.copy(p1), np.copy(p2) # elitism

            if local_state.uniform() < pm : #mutation of childs
                c1 = mutation(c1)

            if local_state.uniform() < pm :
                c2 = mutation(c2)
            newGenome[2 * i - 1] = np.copy(c1)
            newGenome[2 * i] = np.copy(c2)

        newGenome[mu - 1] = np.copy(Genome[np.argmin(fitnessPop)])
        Genome = np.copy(newGenome)

        fitnessPop.clear()
        for i in range(mu):
            fitnessPop.append(fitnessfct(Genome[i], graph))

        eval_cntr += mu
        fmin = np.min(fitnessPop)

        xmin = Genome[:, [np.argmin(fitnessPop)]]
        history.append(fmin)
        if fmin < fcurr_best  :
            fcurr_best = fmin
            xmin = Genome[:, [np.argmin(fitnessPop)]]

        history.append(fcurr_best)
        if np.mod(eval_cntr, int(max_evals / 10)) == 0:
            print(eval_cntr, " evals: fmin=", fmin)
        local_state =  np.random.RandomState(seed + eval_cntr)
        if fmin < 6300:
            print(eval_cntr, " evals: fmin=", fmin, "; done!")
            break

    return xmin,fmin,history


