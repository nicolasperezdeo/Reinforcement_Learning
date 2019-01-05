import numpy as np
import math

class QLearner:

    def __init__(self, num_classes, object_class):
        self.object_class = object_class
        self.num_classes = num_classes
        self.table = np.zeros((num_classes,8,3), dtype=float)
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        self.exploration = 1.0
        self.counter = 0

    def get_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.exploration:
            return math.floor(np.random.uniform(0, 3))
        else:
            return np.argmax(self.table[self.object_class][state])

    def update(self, state, action, next_state, reward):
        old_value = self.table[self.object_class][state][action]
        best_next_value = np.amax(self.table[self.object_class][next_state])*self.discount_factor
        diff = best_next_value - old_value
        update = (diff + reward)*self.learning_rate
        new_value = old_value + update
        self.table[self.object_class][state][action] = new_value
        if self.exploration > 0.0 and self.counter >= 10:
            self.exploration -= 0.1
            self.counter = 0
        else:
            self.counter += 1

    def get_table(self):
        return self.table[self.object_class]

    def get_normalized_table(self):
        table = np.copy(self.table[self.object_class])
        min_val = np.amin(table)
        table += np.abs(min_val)
        max_val = np.amax(table)
        table /= max_val
        return table

    def print_action_sequence(self):
        table = self.get_normalized_table()
        state = 0
        running = True
        while running:
            index = np.argmax(table[state])
            if index == 0:
                print("turn left in state", state)
                state = (state + 1) % 8
            elif index == 1:
                print("classify in state", state)
                running = False
            else:
                print("turn right in state", state)
                state = (state - 1) % 8



