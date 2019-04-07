#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import copy


class Node:
    def __init__(self, matrix, lower_bound, partial_path):
        self.matrix = matrix
        self.lower_bound = lower_bound
        self.partial_path = partial_path

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0):
        pass

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=10.0):
        child_nodes = 1  # number of partial solutions created, 1 for the random tour we set initially
        pruned = 0  # number of nodes that were NOT expanded because they were larger than bssf
        bssf_updates = 0  # number of times we found a better solution
        max_queue_size = 1  # largest the queue ever got
        queue = []  # queue to hold partial solution nodes
        cities = self._scenario.getCities()  # all the cities

        indices = []  # all of the indices, used mostly to calc which cities to expand
        for i in range(len(cities)):
            indices.append(i)

        # get a random solution to use as bssf, this will limit the number of iterations. by not needing to find a solution to start pruning
        bssf = self.defaultRandomTour(time_allowance)['soln']

        # create and reduce the original matrix, put 0 as original cost because this is the first reduction
        m = self.createMatrix(cities)
        m, lowerbound = self.reduceMatrix(m, 0)

        # selecting the first city, create a partial solution node and add it to the queue to start
        heapq.heappush(queue, Node(m, lowerbound, [0]))

        # record start time for time allowance
        start = time.time()

        # while there are still items in the queue and there is still time, pop the smallest cost solution off the queue
        while time.time()-start < time_allowance and len(queue) is not 0:
            node = heapq.heappop(queue)

            # if you already have a better solution, prune this one.
            if node.lower_bound >= bssf.cost:
                pruned += 1
                continue

            # find which cities to visit still
            c_indices = list(set(indices) - set(node.partial_path))

            # if there are no cities to expand, update the bssf, the previous if will already check to make sure this is a better solution
            if len(c_indices) == 0:
                route = []
                for i in node.partial_path:
                    route.append(cities[i])
                bssf = TSPSolution(route)
                bssf.cost = node.lower_bound
                bssf_updates += 1

            # expand each city, create a new node and add it to the queue
            for c in c_indices:
                city = cities[c]
                # index of parent ie the row
                i = node.partial_path[-1]
                # index of the child ie the column
                j = c
                child = self.getChild(node, i, j)
                child_nodes += 1
                if child.lower_bound < bssf.cost:
                    heapq.heappush(queue, child)
                    max_queue_size = max(max_queue_size, len(queue))
                else:
                    pruned += 1

        # return all the results
        results = {}
        results['cost'] = bssf.cost
        results['time'] = time.time() - start
        results['count'] = bssf_updates
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = child_nodes
        results['pruned'] = pruned
        return results

    def createMatrix(self, cities):
        length = len(cities)
        m = np.zeros((length, length))
        for i in range(0, length):
            for j in range(0, length):
                destCity = cities[j]
                srcCity = cities[i]
                m[i][j] = srcCity.costTo(destCity)
        return m

    def reduceMatrix(self, m, cost):
        length = len(m[0])
        # row reduction
        for i in range(length):
            min_cost = np.min(m[i])
            if min_cost == math.inf or min_cost == 0:
                continue
            else:
                cost = cost + min_cost
                for j in range(length):
                    m[i][j] = m[i][j] - min_cost
        # column reduction
        for j in range(length):
            min_cost = np.min(m[:, j])
            if min_cost == math.inf or min_cost == 0:
                continue
            else:
                cost = cost + min_cost
                for i in range(length):
                    m[i][j] = m[i][j] - min_cost
        return m, cost

    def getChild(self, parent, i, j):
        cost = parent.lower_bound
        cost = cost + parent.matrix[i][j]
        m = copy.deepcopy(parent.matrix)

        # set M row i to inf and column j in inf. Also set the child city's route back to the parent to inf
        for x in range(len(m[0])):
            m[i][x] = math.inf
            m[x][j] = math.inf
        m[j][i] = math.inf

        # copy the partial path and add the current city
        m, cost = self.reduceMatrix(m, cost)
        p = copy.deepcopy(parent.partial_path)
        p.append(j)
        child = Node(m, cost, p)
        return child

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        pass
