from Simulator import Simulator
from RLModule import QLearner

object_classes = ["minivan", "iron", "running_shoe", "power_drill", "sunglasses", "hammer", "remote_control", "backpack"]
file_types = ["png", "jpg", "jpg", "jpg", "jpg", "jpg", "jpg", "jpg"]

for i in range(8):
    class_index = i
    # initialize simulator (environment)
    sim = Simulator(object_classes[class_index], file_types[class_index])
    qlearner = QLearner(len(object_classes), class_index)
    print("For", object_classes[class_index], "the best sequence is:")
    for i in range(200):
        state = sim.get_state() #get current state
        action = qlearner.get_action(state) #get action according to policy
        reward = sim.apply_action(action) #apply action
        next_state = sim.get_state() #observe new state
        qlearner.update(state, action, next_state, reward) #update q table
    #print(qlearner.get_normalized_table())
    qlearner.print_action_sequence()
    print()






