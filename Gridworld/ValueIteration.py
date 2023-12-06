# Value Iteration Algorithm
def runValueIteration(states, gamma, theta):
    count = 0
    while True:
        count += 1
        delta = 0
        tempValues = [[0] * 5 for i in range(5)]
        # Run for all states
        for i in range(0, len(states)):
            for j in range(0, len(states[i])):
                state = states[i][j]
                initialValue = state.value
                #bestAction = state.action
                bestValue = 0

                # Take all actions
                for action in range(0, state.actionCount):
                    value = 0

                    # Take all transition states
                    for move in range(0, len(state.transition[action])):
                        [row, column] = state.getNextState(move)
                        value += state.transition[action][move] * (
                                states[row][column].reward + gamma * states[row][column].value)

                    # First action need not be compared
                    if action == 0:
                        bestValue = value
                        bestAction = action

                    # Assign max value to value and arg max to action
                    if value > bestValue:
                        bestAction = action
                        bestValue = value
                #state.action = bestAction
                tempValues[i][j] = bestValue
                finalValue = bestValue

                # Calculate Delta
                delta = max(delta, abs(finalValue - initialValue))
        for i in range(0, len(states)):
            for j in range(0, len(states[i])):
                state = states[i][j]
                state.value = tempValues[i][j]
        if delta < theta:
            break
