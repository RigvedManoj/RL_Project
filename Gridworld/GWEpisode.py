from Gridworld import setInitialState


def runEpisodeGridWorld(states):
    step = 0
    [x, y] = setInitialState()
    currentState = states[x][y]
    while not currentState.checkEndState():
        step += 1
        [newX, newY] = currentState.getNextState()
        newState = states[newX][newY]
        currentState = newState
    return step
