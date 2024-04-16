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
        self.explored_actions = {}
        self.action_index_chosen = None

    def cleanPossibleCalendar(self, possible_calendar):
        # clean out the possible calendar with each of the action contain only relevant events
        A_list = []
        for each_action in possible_calendar:
            current_action = []
            for each_event in each_action:
                if each_event not in self.relevant_events:
                    current_action.append(-1)
                else:
                    current_action.append(each_event)
            A_list.append(current_action)
        return A_list

    def actionSelection(self, possible_calendar, feature_factors):
        A_list = self.cleanPossibleCalendar(possible_calendar)

        d = len(self.relevant_events) * self.number_of_slots
        p_t_a = []  # UCB
        theta_hat = []
        alpha = 0.1

        for action_index in range(len(A_list)):
            if A_list[action_index] not in self.explored_actions.values():
                action_key = len(self.explored_actions)
                self.explored_actions[action_key] = A_list[action_index]
                # TODO: maybe change to list?
                self.A_a_s.append(np.identity(d))
                self.b_a_s.append(np.zeros(d))

            # calculate the machine estimated human preference
            assert np.shape(self.A_a_s[action_index]) == (d, d)
            assert np.shape(self.b_a_s[action_index]) == (d,)
            for key, value in self.explored_actions.items():
                if value == A_list[action_index]:
                    extracted_action_key = key
                    break
            theta_hat.append(np.dot(np.linalg.inv(
                self.A_a_s[extracted_action_key]), self.b_a_s[extracted_action_key]))

            # calculate the UCB
            current_feature_factors = np.array(
                feature_factors[action_index])
            assert np.shape(np.dot(np.linalg.inv(
                self.A_a_s[action_index]), feature_factors[action_index])) == (d,)
            assert np.shape(np.dot(current_feature_factors.T, np.dot(np.linalg.inv(
                self.A_a_s[action_index]), feature_factors[action_index]))) == ()
            p_t_a.append(np.dot(theta_hat[action_index].T, feature_factors[action_index]
                                ) + alpha*np.sqrt(np.dot(current_feature_factors.T, np.dot(np.linalg.inv(self.A_a_s[action_index]), feature_factors[action_index]))))

        print(self.explored_actions)
        # choose the action with the highest UCB
        self.action_index_chosen = np.argmax(p_t_a)
        self.action = A_list[self.action_index_chosen]

        return self.action

    def updateByRating(self, reward_human_rating, possible_calendars, feature_factors):
        feature_chosen = None
        extracted_action_key = None
        d = len(self.relevant_events) * self.number_of_slots
        cleaned_possible_calendars = self.cleanPossibleCalendar(possible_calendars)

        # find the feature of the chosen action
        for action_index in range(len(cleaned_possible_calendars)):
            if cleaned_possible_calendars[action_index] == self.action:
                feature_chosen = feature_factors[action_index]
                break

        #find the index place of the chosen action in A_a_s and b_a_s
        for key, value in self.explored_actions.items():
            if value == self.action:
                extracted_action_key = key
                break

        feature_chosen_reshaped = np.array(feature_chosen).reshape(1,d)
        assert np.shape(feature_chosen * np.array(feature_chosen_reshaped).T) == (d, d)
        self.A_a_s[extracted_action_key] += np.dot(feature_chosen, np.array(feature_chosen_reshaped).T)
        assert np.shape(feature_chosen) == (d,)
        self.b_a_s[extracted_action_key] += reward_human_rating * np.array(feature_chosen)

