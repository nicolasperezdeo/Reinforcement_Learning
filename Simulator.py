from keras.preprocessing.image import load_img
from Classifier import Classifier

class Simulator:

    def __init__(self, object_label, file_type):
        self.state = 0
        self.classifier = Classifier()
        self.object_label = object_label
        self.file_type = file_type
        self.false_classification_cost = 0.5
        self.step_cost = 0.1
        self.debug = False

    def get_state(self):
        return self.state

    def observe_environment(self):
        path = "images/" + self.object_label + "/" + str(self.state) + "." + self.file_type
        return load_img(path, target_size=(224, 224))

    #returns an image according the given action and state
    def apply_action(self, action):
        reward = -self.step_cost
        if action == 0:
            self.state = (self.state + 1) % 8 #rotate left

        elif action == 1:
            image = self.observe_environment()
            predictions = self.classifier.classify(image)[0][0]
            label = predictions[1]
            accuracy = predictions[2]
            if self.object_label == label:
                reward += accuracy
            else:
                reward -= self.false_classification_cost

            if self.debug:
                print(label)

        elif action == 2:
            self.state = (self.state - 1) % 8 #rotate right

        return reward