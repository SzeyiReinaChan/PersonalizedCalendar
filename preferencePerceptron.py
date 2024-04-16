import numpy as np

class PreferencePerceptron:
    # def __init__(self, number_of_features):
    #     self.w = np.zeros(number_of_features)
    # def predict(self, features_list):
    #     predictions = [np.dot(self.w, features) for features in features_list]
    #     return np.argmax(predictions)
    
    # def update(self, chosen_features, true_features):
    #     self.w += (true_features - chosen_features)

#version 2
    def __init__(self, number_of_slots, number_of_features):
        self.w = np.zeros(number_of_slots * number_of_features)

    def phi(self, features_list, action):
        #define the feature representation function phi
        return np.array(features_list[action])
    
    def predict(self, features_list):
        predictions = [np.dot(self.w, features) for features in features_list]
        return np.argmax(predictions)
    
    def update(self, features_list, chosen_features, true_features):
        # self.w += (true_features - chosen_features)
        self.w += self.phi(features_list, true_features) - self.phi(features_list, chosen_features)

    



