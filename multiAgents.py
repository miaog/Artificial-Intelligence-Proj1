# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        def maxi(gameState, ghosts, final, depth):
          if (gameState.isWin() or gameState.isLose()):
            return gameState.getScore()
          b = -(float("inf"))
          actions = gameState.getLegalActions(0);
          for action in actions:
            act = gameState.generateSuccessor(0, action)
            one = mini(act, 1, ghosts, final, depth)
            if (b < one):
              b = one
            if (b == one):
              a = action
          """if you are at the final depth"""
          if (depth == 0): 
            return a
          else:
            return b

        def mini(gameState, ghost, totalghosts, final, depth):
          if (gameState.isLose() or gameState.isWin()):
            return gameState.getScore()
          w = float("inf")
          actions = gameState.getLegalActions(ghost);
          if (ghost == totalghosts):
            for action in actions:
              if (depth == final):
                w = min(w, self.evaluationFunction(gameState.generateSuccessor(ghost, action)))
              else:
                w = min(w, maxi(gameState.generateSuccessor(ghost, action), totalghosts, final, depth + 1))
          else:
            for action in actions:
              if (ghost == totalghosts):
                if (depth == final):
                  w = min(w, self.evaluationFunction(gameState.generateSuccessor(ghost, action)))
                next = 0
              g = gameState.generateSuccessor(ghost, action)
              w = min(w, mini(g, ghost + 1, totalghosts, final, depth))
          """if you are at the final depth"""
          if (depth == 0):
            return w
          else:
            return w
        agents = gameState.getNumAgents()
        final = self.depth - 1
        return maxi(gameState, agents - 1, final, 0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def maxi(gameState, ghosts, final, depth):
          if (gameState.isWin() or gameState.isLose()):
            return gameState.getScore()
          b = -(float("inf"))
          actions = gameState.getLegalActions(0);
          for action in actions:
            act = gameState.generateSuccessor(0, action)
            one = mini(act, 1, ghosts, final, depth)
            if (b < one):
              b = one
            if (b == one):
              a = action
          """if you are at the final depth"""
          if (depth == 0): 
            return a
          else:
            return b

        def mini(gameState, ghost, totalghosts, final, depth):
          if (gameState.isLose() or gameState.isWin()):
            return gameState.getScore()
          w = 0.0
          i = 0.0
          actions = gameState.getLegalActions(ghost);
          if (ghost == totalghosts):
            """if last ghost"""
            for action in actions:
              i += 1.0
              if (depth == final):
                w += float(self.evaluationFunction(gameState.generateSuccessor(ghost, action)))
              else:
                w += float(maxi(gameState.generateSuccessor(ghost, action), totalghosts, final, depth + 1))
          else:
            """if not last ghost"""
            for action in actions:
              i += 1.0
              if (ghost == totalghosts):
                if (depth == final):
                  w += float(self.evaluationFunction(gameState.generateSuccessor(ghost, action)))
                next = 0
              w += float(mini(gameState.generateSuccessor(ghost, action), ghost + 1, totalghosts, final, depth))
          """if you are at the final depth"""
          if (depth == 0):
            p = float(w)/ float(i)
            return p
          else:
            p = float(w)/ float(i)
            return p

        agents = gameState.getNumAgents()
        final = self.depth - 1
        return maxi(gameState, agents - 1, final, 0)

    def maxValue(self, gameState, depth):

      x = 10
      v = -100000
      bestAction = ""

      for action in gameState.getLegalActions(0):
            if action != Directions.STOP:
                x = self.expValue(gameState.generateSuccessor(0, action), depth, 1)

                if x > v:
                    v = x
                    bestAction = action

      if depth == 0:
            return bestAction
      else:
            return v  

    def expValue(self, gameState, depth, numGhosts):
        
        if depth == self.depth - 1:
            return self.evaluationFunction(gameState)

        v = 100000  

        #l = len(gameState.getLegalActions(gameState)
        print gameState.getLegalActions(gameState)
        
        for action in gameState.getLegalActions():
            if action != Directions.STOP:
              if numGhost != gameState.getNumAgents()-1:
                v = min(v, self.maxValue(gameState.generateSuccessor(numGhost, action), depth + 1))
                print "BLA"

            else:
              v = min(v, self.expValue(gameState.generateSuccessor(numGhost, action), depth, numGhost + 1))
              print "BLA"  
                
        return v   

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 9).

      DESCRIPTION: For betterEvaluationFunction, we had to debate about what are the most
      important factors that helps Pacman win or lose. We ranked our decisions and these are 
      what we came up with in descending order which you will see in our code, but first we 
      need to innitialize some lists and variables which are pretty self explanatory but the
      most important one to keep note of is our variable, 'final', which holds the currentGameState
      score initially and after going through all the factors, becomes what we return at the end. 
      Now here is our list of factors that we based our function off of:
        1. We check first if the currentGameState will guarantee a win or lose. If it is a win,
          then we return float("inf") to maximize the incentive to pick that option and if the
          game state guarantees a loss, then we return -float("inf") to maximize the incentive
          NOT to pick that route
        2. The second most important thing is making sure Pacman stays away from the ghosts. To make
          sure that Pacman does this we subtract the closest ghost from Pacman multiplied by 10 to
          heighten this importance
        3. The third important factor is the food. We want to make sure of course that we eat all
          the food and we get this by getting the closest food to Pacman and subtracting it
          from 'final'
        4. The last most important factor is making sure you eat the capsules. We want to make sure
          that if there are capsules still on the board, then we want to make sure we eat them. 
          To make an incentive we subtract it from 'final' to make sure Pacman clears the capsules
    """
    def getDistance(a, b):
      c = abs(a[0] - b[0])
      d = abs(a[1] - b[1])
      f = c + d
      return f
    pacman = list(currentGameState.getPacmanPosition())
    closestFood = []
    ghostnums = []
    closestGhost = float("inf")
    final = currentGameState.getScore()
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getNumAgents() - 1
    i = 1
    while i <= ghosts:
      ghostnums.append(i)
      i += 1
    capsuleList = currentGameState.getCapsules() #len(capsuleList) --> number of capsules

    if currentGameState.isWin():
      return float("inf")
    if currentGameState.isLose():
      return -float("inf")
    
    for ghost in ghostnums:
      nextdist = getDistance(list(currentGameState.getGhostPosition(ghost)), pacman)
      closestGhost = min(closestGhost, nextdist)
    final -= (10 * closestGhost)

    for food in foods:
        a = getDistance(food, pacman)
        closestFood.append(a)  
    final -= min(closestFood)
    final -= len(capsuleList)

    return final

# Abbreviation
better = betterEvaluationFunction

