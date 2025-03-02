# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            newValues = util.Counter()

            for state in self.mdp.getStates():
                # skip if state is terminal
                if self.mdp.isTerminal(state):
                    continue
                
                # get max Q value
                bestValue = float('-inf')

                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    bestValue = max(bestValue, qValue)

                newValues[state] = bestValue

            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # calcualte Q value by summation of all possible next states
        qValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for nxtState, prob in transitions:
            reward = self.mdp.getReward(state, action, nxtState)
            qValue += prob * (reward + self.discount * self.values[nxtState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        # get action with highest Q value
        bestAction = None
        bestScore = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestScore:
                bestScore = qValue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        n = len(states)

        for i in range(self.iterations):
            state = states[i % n]
            
            # if terminal, skip
            if self.mdp.isTerminal(state):
                continue
            
            bestValue = float('-inf')

            for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                bestValue = max(bestValue, qValue)

            self.values[state] = bestValue
            
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = collections.defaultdict(set)
        states = self.mdp.getStates()

        # compute predecessors of all states
        for state in states:
            if self.mdp.isTerminal(state):
                continue

            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        # initialize priority queue
        pq = util.PriorityQueue()

        # for each non-terminal state s, do .....
        for state in states:
            if self.mdp.isTerminal(state):
                continue

            bestValue = float('-inf')

            for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                bestValue = max(bestValue, qValue)

            diff = abs(self.values[state] - bestValue)
            pq.update(state, -diff)

        # for iternation in 0,1,2...., self.iterations, do:....
        for i in range(self.iterations):
            # if the priority queue is empty, then terminate
            if pq.isEmpty():
                break

            # pop a state s off the priority queue
            state = pq.pop()

            # update s's value
            if not self.mdp.isTerminal(state):
                bestValue = float('-inf')

                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    bestValue = max(bestValue, qValue)

                self.values[state] = bestValue

            # for each predecessor p of s, do...
            for p in predecessors[state]:
                bestValue = float('-inf')

                # find the highest q value
                for action in self.mdp.getPossibleActions(p):
                    qValue = self.computeQValueFromValues(p, action)
                    bestValue = max(bestValue, qValue)

                diff = abs(self.values[p] - bestValue)

                # if diff > theta, then push p into the priority queue with priority -diff
                if diff > self.theta:
                    pq.update(p, -diff)