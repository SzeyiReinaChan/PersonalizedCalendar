import numpy as np

# Algorithm: LinearUCB
# Paper: A Contextual-Bandit Approach to Personalized News Article Recommendation


class LinearUCB:
    def __init__(self, relevant_events, number_of_slots):
        self.relevant_events = relevant_events
        self.number_of_slots = number_of_slots
        # A_a_s: store all the A_a for each action
        self.A_a_s = []
        # b_a_s: store all the b_a for each action
        self.b_a_s = []
        self.action_index_chosen = None

    def actionSelection(self, possible_calendar, feature_factors):
        A_list = possible_calendar
        d = len(self.relevant_events) * self.number_of_slots
        # p_t_a: UCB
        p_t_a = []
        theta_hat = []
        alpha = 0.1
        for action_index in range(len(A_list)):
            if A_list[action_index] not in self.A_a_s:  # TODO
                self.A_a_s.append(np.identity(d))
                self.b_a_s.append(np.zeros(d))

            # calculate the machine estimated human preference
            assert np.shape(self.A_a_s[action_index]) == (d, d)
            assert np.shape(self.b_a_s[action_index]) == (d,)
            theta_hat.append(np.dot(np.linalg.inv(
                self.A_a_s[action_index]), self.b_a_s[action_index]))  # TODO

            # calculate the UCB
            current_feature_factors = np.array(
                feature_factors[action_index])
            assert np.shape(np.dot(np.linalg.inv(
                self.A_a_s[action_index]), feature_factors[action_index])) == (d,)
            assert np.shape(np.dot(current_feature_factors.T, np.dot(np.linalg.inv(
                self.A_a_s[action_index]), feature_factors[action_index]))) == ()
            p_t_a.append(np.dot(theta_hat[action_index].T, feature_factors[action_index]
                                ) + alpha*np.sqrt(np.dot(current_feature_factors.T, np.dot(np.linalg.inv(self.A_a_s[action_index]), feature_factors[action_index]))))

        # choose the action with the highest UCB
        self.action_index_chosen = np.argmax(p_t_a)
        self.action = A_list[self.action_index_chosen]

        return self.action

    def updateByRating(self, reward_human_rating, feature_factors):
        d = len(self.relevant_events) * self.number_of_slots
        # update A_a and b_a
        feature_chosen = np.array(
            feature_factors[self.action_index_chosen])  # TODO
        # assert np.shape(feature_chosen.T *
        #                 feature_factors[self.action_index_chosen]) == (d, d)
        self.A_a_s[self.action_index_chosen] += feature_chosen.T * \
            feature_factors[self.action_index_chosen]  # TODO
        assert np.shape(feature_chosen) == (d,)
        self.b_a_s[self.action_index_chosen] += reward_human_rating * \
            feature_chosen
