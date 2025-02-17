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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        foodList = newFood.asList()

        # get distance to closest food
        if foodList:
            minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            score += 5 / (minFoodDistance)
        else: 
            minFoodDistance = 1
            score += 5 / (minFoodDistance)

        # get distance to closest ghost
        minGhostDistance = min(manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates)
        # if the ghost is very close to pacman, it should be inventivized to run away
        if minGhostDistance < 2:
            score -= 200

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

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
    Your minimax agent (question 2)
    """

    # get the max value for pacman
    def maxValue(self, state, agentIndex, nextAgent, newDepth):
        value = float('-inf')

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            value = max(value, self.miniMax(successor, nextAgent, newDepth))

        return value

    # get the min value for the ghosts  
    def minValue(self, state, agentIndex, nextAgent, newDepth):
        value = float('inf')

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            value = min(value, self.miniMax(successor, nextAgent, newDepth))

        return value

    def miniMax(self, state, agentIndex, currentDepth):
        # stop if state is winning, losing, or at max depth
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)

        numAgents = state.getNumAgents()  # Total number of agents (Pacman + ghosts)
        nextAgent = (agentIndex + 1) % numAgents
        newDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

        # choose max value for pacman
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, nextAgent, newDepth)
        # choose max value for pacman
        else:
            return self.minValue(state, agentIndex, nextAgent, newDepth)

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # pacman will then choose the best option
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.miniMax(successor, 1, 0) 
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    
    # max value for pacman
    def maxValue (self, state, agentIndex, nextAgent, newDepth, alpha, beta):
        value = float('-inf')

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            value = max(value, self.alphaBeta(successor, nextAgent, newDepth, alpha, beta))
            if value > beta: 
                return value
            alpha = max(alpha, value)

        return value 

    # min value for ghosts
    def minValue (self, state, agentIndex, nextAgent, newDepth, alpha, beta):
        value = float('inf')

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            value = min(value, self.alphaBeta(successor, nextAgent, newDepth, alpha, beta))
            if value < alpha: 
                return value
            beta = min(beta, value)

        return value 

    # alpha beta implementation
    def alphaBeta(self, state, agentIndex, currentDepth, alpha, beta):
        # Check terminal conditions (win/lose state or max depth reached)
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        
        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        newDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

        # choose max value for pacman
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, nextAgent, newDepth, alpha, beta)
        
        # choose max value for pacman
        else:
            return self.minValue(state, agentIndex, nextAgent, newDepth, alpha, beta)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.alphaBeta(successor, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, state, agentIndex, currentDepth):
        # check is state is a terminal state or max depth is reached
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        newDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

        # choose max value for pacman
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, nextAgent, newDepth)
            
        # choose expercted value for ghosts
        else:
            return self.expValue(state, agentIndex, nextAgent, newDepth)

    # max value for pacman
    def maxValue(self, state, agentIndex, nextAgent, newDepth):
        v = float('-inf')

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successor, nextAgent, newDepth))

        return v

    # expected value for ghosts
    def expValue(self, state, agentIndex, nextAgent, newDepth):
        v = 0
        p = 1.0 / len(state.getLegalActions(agentIndex))  # probabilty

        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            v += p * self.value(successor, nextAgent, newDepth)

        return v
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"    
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.value(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    use current game score
    use distance to closest food
    use distance to closest ghost
    penalize based on how much food remains on the map
    chase ghosts if they are scared
    chase pellets if there are any on the map
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    capsules = currentGameState.getCapsules()
    
    # get distance to closest food
    foodScore = 0
    foodList = newFood.asList()
    if foodList:
        minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
        foodScore += 5 / (minFoodDistance)
    else: 
        minFoodDistance = 1
        foodScore += 5 / (minFoodDistance)
    
    # penalize based on how much food remains on the map
    foodPenalty = 5 * len(foodList)
    
    # get distance to closest ghost
    ghostScore = 0
    for ghost in newGhostStates:
        ghostDistance = manhattanDistance(newPos, ghost.getPosition())
        # if ghost in scared state, pacman should chase it
        if ghost.scaredTimer == 0:
            ghostScore += 400 
        # if the ghost is very close to pacman, it should be inventivized to run away
        else:
            if ghostDistance < 2:
                ghostScore -= 200 

    # chase pellets if there are any on the map
    capsuleScore = 0
    if capsules:
        minCapsuleDistance = min(manhattanDistance(newPos, capsule) for capsule in capsules)
        capsuleScore += 100 / (minCapsuleDistance)
    else:
        minCapsuleDistance = 1
        capsuleScore += 100 / (minCapsuleDistance)
    
    # add all scores together
    totalScore = score + foodScore - foodPenalty + ghostScore + capsuleScore
    return totalScore

# Abbreviation
better = betterEvaluationFunction
