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
        states = self.mdp.getStates()  # get all states
        for it in range(self.iterations):  # for each iteration
            new_values = self.values.copy()  # copy the previous values
            for state in states:  # for each state
                max_value = None  # set a max value to be updated later
                actions = self.mdp.getPossibleActions(state)  # get all actions
                for action in actions:  # for each action
                    temp_value = self.computeQValueFromValues(state, action)  # compute q-val
                    if max_value is None or max_value < temp_value:  # update max-value
                        max_value = temp_value
                if max_value is None:  # extra check for resetting max_value
                    max_value = 0
                new_values[state] = max_value
            self.values = new_values  # update self.values


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
        q_value = 0  # initialize q val
        transition_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)  # get transition function
        for state_prob in transition_states_probs:
            reward = self.discount * self.values[state_prob[0]] \
                     + self.mdp.getReward(state, action, state_prob[0])  # combine reward from discount and reward
            q_value += state_prob[1] * reward  # add reward to existing state value to get q_value
        return q_value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:  # if no possible actions, immediately return
            return None
        max_action = None  # set max values for all actions and values
        max_value = None

        for action in actions:
            value = self.computeQValueFromValues(state, action)
            if max_value is None or max_value < value:  # compare values
                max_value = value
                max_action = action  # set associated value and action to the max
        return max_action  # return the maximum value action

        # util.raiseNotDefined()

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
        states = self.mdp.getStates()  # getting states
        it_mod = 0  # initializing iterator
        for it in range(self.iterations):
            if it_mod == len(states):  # if states is the same length as current iterator, reset iterator to 0
                it_mod = 0
            state = states[it_mod]  # get the relevant state
            it_mod += 1  # increment iterator
            if not self.mdp.isTerminal(state):  # if the state is not terminal
                actions = self.mdp.getPossibleActions(state)  # get possible actions
                q_values = []
                for action in actions:
                    q_values.append(self.getQValue(state, action))  # add q values to possible actions list
                self.values[state] = max(q_values)  # set this state to the highest possible q value

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
        predecessors = {}  # initializing predecessors
        states = self.mdp.getStates()  # getting states

        for state in states:  # for each state
            predecessors[state] = set()  # if not terminal, create set
        for state in states:  # for each state
            actions = self.mdp.getPossibleActions(state)  # get all actions
            for action in actions:  # for each action
                transition_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
                for (next_state, prob) in transition_states_probs:
                    if prob > 0:  # if the second component of the tuple is more than 0
                        predecessors[next_state].add(state)  # add the state to predecessors

        queue = util.PriorityQueue()  # initializing priority queue

        for state in states:
            if not self.mdp.isTerminal(state):  # if not terminal
                current_val = self.getValue(state)  # get value
                action = self.computeActionFromValues(state)  # get best action
                q_val = self.computeQValueFromValues(state, action)  # get best actions q_value
                absolute_difference = abs(current_val - q_val)  # find difference
                queue.push(state, -absolute_difference)  # push state to queue with priority of -absolute_diff
        for it in range(self.iterations):  # iterate for number of iterations
            if queue.isEmpty():
                return
            state = queue.pop()  # get the highest priority state

            action = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, action)  # update values to match it's q-val

            for predecessor in predecessors[state]:  # updating predecessors
                current_val = self.values[predecessor]
                action_temp = self.computeActionFromValues(predecessor)
                q_val = self.computeQValueFromValues(predecessor, action_temp)
                absolute_difference = abs(current_val - q_val)
                if absolute_difference > self.theta:  # if more than theta, update the corresponding queue item
                    queue.update(predecessor, -absolute_difference)





