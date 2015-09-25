# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    visited = []
    result = []
    fronteir = util.Queue()
    fall = []
    result = [[]]
    a = problem.getStartState()
    fr = [a]
    fronteir.push([a, []])

    while not fronteir.isEmpty():
        vertex, r = fronteir.pop()

        # print vertex
        if (problem.goalTest(vertex)):
            return r
        for action in problem.getActions(vertex):
            child = problem.getResult(vertex, action)
            if (child not in fr):
                if (child not in visited):
                    if (problem.goalTest(child)):
                        return r + [action]
                    fronteir.push([child, r + [action]])
                    fr.append(child)
        visited.append(vertex)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

good = 0;

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth.

    Begin with a depth of 1 and increment depth by 1 at every step.
    """
    global good
    depth = 0;
    a = problem.getStartState()
    while (True):
        result = depthlimitedsearch(problem, a, depth, [], [], [[a, 0]], [[]])
        if (good == 1):
            good = 0
            return result
        depth = depth + 1

def depthlimitedsearch(problem, a, dept, res, vis, fro, act):
    global good
    visited = vis
    result = res
    fronteir = fro
    fall = []
    result = act
    fr = [a]

    if (dept == 0):
        if (problem.goalTest(a)):
            good = 1
        return result
    while (fronteir):
        v = fronteir.pop()
        fr.pop()
        depth = v.pop()
        vertex = v.pop()
        r = result.pop()
        if (problem.goalTest(vertex)):
            good = 1
            return r
        if depth < dept:
            for action in problem.getActions(vertex):
                child = problem.getResult(vertex, action)
                if (child not in fr):
                    if (child not in visited):
                        p = v[:]
                        p.extend([vertex, child, depth + 1])
                        fronteir.append(p)
                        fr.append(child)
                        l = r[:]
                        l.extend([action])
                        result.append(l)
        visited.append(vertex)
    return result


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue() # Orders nodes by combined cost and heuristic
    start = problem.getStartState() # Start is the beginning node in the problem
    frontier.push(start, heuristic(start, problem)) # Put the start node on the frontier
    node_parent = {} # Tracks where each node came from
    state_costs = {} # Tracks the total cost of states so far
    state_costs[start] = 0 # The total cost for the start node is 0
    state_action = {} # This maps a state to an action that produced the state
    final_path = [] # This is the list of actions that we return at the end
    
    while frontier.isEmpty() == False:
        state = frontier.pop()
        if problem.goalTest(state):
            this_action = state_action[state]
            final_path.append(this_action)
            previous = node_parent[state]; 
            while previous != start:
                this_action = state_action[previous]
                final_path.append(this_action)
                previous = node_parent[previous];
            final_path = final_path[::-1]
            return final_path
        for action in problem.getActions(state):
            resulting_state = problem.getResult(state, action)
            g = state_costs[state] + problem.getCost(state, action)
            if resulting_state not in state_costs or g < state_costs[resulting_state]:
                h = heuristic(resulting_state, problem)
                f = g + h
                frontier.push(resulting_state, f)
                state_costs[resulting_state] = g
                node_parent[resulting_state] = state
                state_action[resulting_state] = action

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
