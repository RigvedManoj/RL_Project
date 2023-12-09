import math

import numpy
import random


class GridWorldState:
    def __init__(self, state, actionCount, stateType="Normal"):
        self.state = state
        self.stateType = stateType
        self.actionCount = actionCount
        self.actionProbabilities = [0.25, 0.25, 0.25, 0.25]
        self.transition = [None] * actionCount
        self.value = 0
        self.oldValue = 0
        self.qValue = [10, 10, 10, 10]
        self.visits = [0, 0, 0, 0]
        self.visited = [False, False, False, False]
        self.reward = 0
        self.Model = [{}, {}, {}, {}]  # next_state:visitCount for each state.action
        self.parent = None
        self.parentAction = None

    def __lt__(self, other):
        return (self.state[0] + self.state[1]) < (other.state[0] + other.state[1])

    # left(0),up(1),down(2),right(3) (policies are stochastic)
    def takeAction(self):
        return numpy.random.choice(numpy.arange(0, 4), p=self.actionProbabilities)

    def setActionProbabilities(self, epsilon):
        maxQ = max(self.qValue)
        optimalAction = 0
        for action in range(0, 4):
            if self.qValue[action] == maxQ:
                optimalAction += 1
        for action in range(0, 4):
            if self.qValue[action] == maxQ:
                self.actionProbabilities[action] = (1 - epsilon) / optimalAction + epsilon / 4
            else:
                self.actionProbabilities[action] = epsilon / 4

    def setActionUCB(self, c):
        t = 1
        for action in range(0, self.actionCount):
            t += self.visits[action]
        value = 0
        bestAction = 0
        for action in range(0, self.actionCount):
            numerator = math.log(t, math.e)
            denominator = self.visits[action] + 1
            value = max(value, self.qValue[action] + c * math.sqrt(numerator / denominator))
            self.actionProbabilities[action] = 0
            bestAction = action
        self.actionProbabilities[bestAction] = 1

    def setTransition(self, actions, stateProbLists):
        for action in actions:
            self.transition[action] = stateProbLists[action]

    # [left,up,down,right,same] -> [0,1,2,3,4]
    def getNextState(self, action, move=None):
        if move is None:
            move = numpy.random.choice(numpy.arange(0, 5), p=self.transition[action])
        newX = self.state[0]
        newY = self.state[1]

        if move == 0:  # Left Logic
            newY = self.state[1] - 1
        elif move == 1:  # Up Logic
            newX = self.state[0] - 1
        elif move == 2:  # Down Logic
            newX = self.state[0] + 1
        elif move == 3:  # Right Logic
            newY = self.state[1] + 1

        # Check if position goes outside grid
        if newX < 0 or newX > 4:
            newX = self.state[0]
        if newY < 0 or newY > 4:
            newY = self.state[1]

        # Check Obstacle Space
        if newX == 2 and newY == 2:
            newX = self.state[0]
            newY = self.state[1]
        if newX == 3 and newY == 2:
            newX = self.state[0]
            newY = self.state[1]

        return [newX, newY]

    # Set Reward
    def setReward(self, reward):
        self.reward = reward

    # Check EndState
    def checkEndState(self):
        if self.stateType == "EndState":
            return True
        else:
            return False


# Set all states by defining policies, transitions and rewards.
def createGridworld():
    states = []
    for i in range(0, 5):
        tempStates = []
        for j in range(0, 5):
            if i != 4 or j != 4:
                # Set Obstacles
                if (i == 2 or i == 3) and j == 2:
                    s = GridWorldState([i, j], 0, "Obstacles")
                    s.setReward(0)
                    s.qValue = [0, 0, 0, 0]
                    tempStates.append(s)
                    continue
                # Set all other States
                s = GridWorldState([i, j], 4)
                transitionLeft = [0.8, 0.05, 0.05, 0, 0.1]  # left,up,down,right,same
                transitionUp = [0.05, 0.8, 0, 0.05, 0.1]  # left,up,down,right,same
                transitionDown = [0.05, 0, 0.8, 0.05, 0.1]  # left,up,down,right,same
                transitionRight = [0, 0.05, 0.05, 0.8, 0.1]  # left,up,down,right,same
                s.setTransition([0, 1, 2, 3], [transitionLeft, transitionUp, transitionDown, transitionRight])
                s.setReward(0)
                # Set reward for Water State
                if i == 4 and j == 2:
                    s.setReward(-10)
                tempStates.append(s)
            # Set Goal State
            else:
                s = GridWorldState([i, j], 0, "EndState")
                s.setReward(10)
                s.qValue = [0, 0, 0, 0]
                tempStates.append(s)
        states.append(tempStates)
    return states


# randomly choose an initial state with equal probability
def setInitialState():
    while True:
        i = random.randint(0, 4)
        j = random.randint(0, 4)
        if (i == 2 or i == 3) and j == 2:
            continue
        if i == 4 and j == 4:
            continue
        else:
            break
    return [i, j]
