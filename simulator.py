import numpy as np
import random
import copy
from linearUCB import LinearUCB
import matplotlib.pyplot as plt
np.random.seed(0)
random.seed(0)

# some features for future use
# D_day = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
#          "Thursday": 4, "Friday": 5, "Saturday": 6}
# D_start = ["12:00am", "11:59pm"]
# D_duration = [i for i in range(24*60+1)]
# D_periods = {"morning": 0, "afternoon": 1, "evening": 2}

irrelevant_events = ["gym", "class"]
relevant_events = ["assignment_out", "reading"]
events = irrelevant_events + relevant_events

assignment_out_constrain = [0, 1, 2]
reading_constrain = [0, 1, 2, 3, 4]

# Total number of slots (5 for now for simplicity)
# Will be changed to 24*60/30 = 48 for 30 minutes slots for day implementation
number_of_slots = 5


# For generating the preference for each event in each slot.
def preferenceGenerator(calendar):
    preference_count = 0
    for each in calendar.values():
        if each in relevant_events:
            preference_count += 1
    print("preference_count=", preference_count)

    # randomly generated θ(preference) for evaluation, e.g., θ is uniformly random
    preference = np.random.uniform(0, 1, preference_count*number_of_slots)

    return preference


def featureListGenerator(input, num_range):
    features_list = []
    for each in relevant_events:
        for i in range(*num_range):
            # print("each, i: ", each, i)
            if input[i] == each:
                features_list.append(1)
            else:
                features_list.append(0)
    return features_list


def get_irrelevant_calendar():
    st_irrelevant_calendar = [-1]*number_of_slots  # empty calendar
    for each in irrelevant_events:
        # allow 1 in max for each irrelevant event
        num_each = np.random.randint(0, 2, size=1)[0]
        for _ in range(num_each):
            # randomly assign the position for the irrelevant events
            probability = np.random.uniform(0, 1, number_of_slots)
            probability_normalized = probability / probability.sum()
            each_position = np.random.choice(
                [0, 1, 2, 3, 4], p=probability_normalized)
            print("each_position=", each_position)

            if st_irrelevant_calendar[each_position] == -1:
                st_irrelevant_calendar[each_position] = each
            else:
                # if there is already an event, skip
                print("Skipped placement")
                continue
        print("st_irrelevant_calendar=", st_irrelevant_calendar)
    return st_irrelevant_calendar


def eligibleActionsGenerator(st_irrelevant_calendar):
    slots_available = [i for i in range(
        number_of_slots) if st_irrelevant_calendar[i] == -1]
    print("slots_available=", slots_available)

    possible_calendars = []
    for assignment_out_slot in slots_available:
        for reading_slot in slots_available:
            if reading_slot == assignment_out_slot:
                continue
            if assignment_out_slot not in assignment_out_constrain:
                continue
            if reading_slot not in reading_constrain:
                continue
            new_calendar = copy.copy(st_irrelevant_calendar)
            new_calendar[assignment_out_slot] = "assignment_out"
            new_calendar[reading_slot] = "reading"
            possible_calendars.append(new_calendar)
    print("============Possible Actions============")
    for i in possible_calendars:
        print(i)
    return slots_available, possible_calendars

def plot_data(dataset_1, dataset_2, title, plot_filename):
    # reward = dataset_1
    # regret = dataset_2
    # plt.plot(reward)
    # plt.plot(regret) 
    # plt.xlabel('rounds')
    # plt.ylabel('reward value')
    # plt.title(title)
    # plt.legend(['reward', 'regret'], loc='upper left')
    # plt.savefig(plot_filename)
    # plt.close()


    pp = dataset_1
    ucb = dataset_2
    plt.plot(pp)
    plt.plot(ucb)
    
    plt.xlabel('rounds')
    plt.ylabel('reward value')
    plt.title(title)
    plt.legend(['pp', 'ucb'], loc='upper left')
    plt.savefig(plot_filename)
    plt.close()


def simulation(rounds=2):
    # initialize_feature = featureListGenerator(
    #     calendar, [1, number_of_slots + 1])
    # print("============**Init**============")
    # print("initialize_feature= ", initialize_feature)

    # simulate preference
    # preference = preferenceGenerator(calendar)
    preference = np.random.uniform(
        0, 1, len(relevant_events)*number_of_slots)
    print("============Preference============")
    print("preference=", preference)

    # assert (len(preference) == len(initialize_feature))

    # initialize the algorithm
    linear_ucb = LinearUCB(relevant_events, number_of_slots)

    # initialize reward
    ucb_total_reward = 0.0
    pp_total_reward = 0.0
    pp_total_feedback_reward = 0.0
    pp_total_regret = 0.0

    pp_total_reward_dataset = []
    pp_total_feedback_reward_dataset = []
    pp_total_regret_dataset = []

    #initialize weights for perceptron
    num_features = number_of_slots * len(relevant_events)
    w = np.zeros(num_features)


    # Rounds of simulations
    for t in range(rounds):
        print("\n============ Round", t, "============")
        st_irrelevant_calendar = get_irrelevant_calendar()
        slots_available, possible_calendars = eligibleActionsGenerator(
            st_irrelevant_calendar)

        # LinearUCB simulation===================================
        # generate the feature factors for each possible calendar
        feature_factors = []
        for each_possible_calendar in possible_calendars:
            feature_factors.append(featureListGenerator(
                each_possible_calendar, [number_of_slots]))

        ucb_chosen_action_at = linear_ucb.actionSelection(
            possible_calendars, feature_factors)
        print("action=", ucb_chosen_action_at)

        # get the reward - reward of the true human preference
        chosen_action_feature_list = featureListGenerator(
            ucb_chosen_action_at, [number_of_slots])
        ucb_human_reward = np.dot(preference, chosen_action_feature_list)
        print("human reward=", ucb_human_reward)

        # update the reward based on the human rating + random simulated noise
        ucb_noisy_reward = ucb_human_reward + \
            np.random.normal(loc=0.0, scale=0.1)
        print("noisy reward=", ucb_noisy_reward)

        # update the algorithm based on the reward
        linear_ucb.updateByRating(ucb_noisy_reward, feature_factors)

        # for evaluation purpose
        ucb_total_reward += ucb_noisy_reward
        print("\ntotal_reward=", ucb_total_reward)

        # Preference Perceptron simulation===================================
        best_action = None
        max_val = float('-inf')
        for calendar in possible_calendars:
            feature_vector = featureListGenerator(calendar, [number_of_slots])
            val = np.dot(w, feature_vector)
            if val > max_val:
                max_val = val
                best_action = calendar
                
        #Get the reward for the best action based on Perceptron
        pp_reward = np.dot(preference, featureListGenerator(best_action, [number_of_slots]))
        print("Reard for best action based on Perceptron: ", pp_reward)
        pp_total_reward += pp_reward
        print("\npp total_reward=", pp_total_reward)

        #Obtain feedback by selecting from calendars with rewards greater than or equal to the reward
        feedback_calendars = []
        for calendar in possible_calendars:
            calandar_reward = np.dot(preference, featureListGenerator(calendar, [number_of_slots]))
            if calandar_reward >= pp_reward:
                feedback_calendars.append(calendar)
        if not feedback_calendars:
            feedback_calendars.append(best_action)
        
        #Randomly select feedback from calandars with rewards greater than or equal to the reward
        feedback_index = np.random.choice(len(feedback_calendars))
        feedback = feedback_calendars[feedback_index]
        feedback_reward = np.dot(preference, featureListGenerator(feedback, [number_of_slots]))
        pp_total_feedback_reward += feedback_reward
        print("\npp total_feedback_reward=", pp_total_feedback_reward)

        pp_total_regret += abs(pp_reward - feedback_reward)
        print("\npp total_regret=", pp_total_regret)

        pp_total_reward_dataset.append(pp_total_reward)
        pp_total_feedback_reward_dataset.append(pp_total_feedback_reward)
        pp_total_regret_dataset.append(pp_total_regret)

        #Update the weights based on the feedback
        phi_best_action = np.array(featureListGenerator(best_action, [number_of_slots]))
        phi_feedback = np.array(featureListGenerator(feedback, [number_of_slots]))

        w += phi_feedback - phi_best_action

    print("Final weights after simulation:", w)
    plot_data(pp_total_feedback_reward, ucb_total_reward, "pp vs ucb", "pp.png")

        # action_index = preference_perceptron.predict(feature_factors)
        # chosen_action = possible_calendars[action_index]
        # chosen_action_features = feature_factors[action_index]

        # #Get the true best action according to the preference
        # #TODO: Implement the function get_true_best_action
        # true_action_index = get_true_best_action()
        # true_action = possible_calendars[true_action_index]
        
        # reward = np.dot(preference_perceptron.weights, chosen_action_features)
        # preference_perceptron.update(feature_factors, action_index, true_action_index)
        # pp_total_reward += reward

        # print("\npp total_reward=", pp_total_reward)

        # 1. Get st=irrelevant calendar
        # st_irrelevant_calendar = [-1]*number_of_slots  # empty calendar
        # for each in irrelevant_events:
        #     # allow 1 in max for each irrelevant event
        #     num_each = np.random.randint(0, 2, size=1)[0]
        #     for _ in range(num_each):
        #         # randomly assign the position for the irrelevant events
        #         probability = np.random.uniform(0, 1, number_of_slots)
        #         probability_normalized = probability / probability.sum()
        #         each_position = np.random.choice(
        #             [0, 1, 2, 3, 4], p=probability_normalized)
        #         print("each_position=", each_position)

        #         if st_irrelevant_calendar[each_position] == -1:
        #             st_irrelevant_calendar[each_position] = each
        #         else:
        #             # if there is already an event, skip
        #             continue
        #     print("st_irrelevant_calendar=", st_irrelevant_calendar)

        # 2. Find all eligible actions
        # E.g., constraint: in this week, both assignment_out and reading have to be assigned.
        # assignment_out has to be slot 0,1,2. reading can be anytime.
        # slots_available = [i for i in range(
        #     number_of_slots) if st_irrelevant_calendar[i] == -1]
        # print("slots_available=", slots_available)

        # possible_calendars = []
        # for assignment_out_slot in slots_available:
        #     for reading_slot in slots_available:
        #         if reading_slot == assignment_out_slot:
        #             continue
        #         if assignment_out_slot not in assignment_out_constrain:
        #             continue
        #         if reading_slot not in reading_constrain:
        #             continue
        #         new_calendar = copy.copy(st_irrelevant_calendar)
        #         new_calendar[assignment_out_slot] = "assignment_out"
        #         new_calendar[reading_slot] = "reading"
        #         possible_calendars.append(new_calendar)
        # print("============Possible Actions============")
        # for i in possible_calendars:
        #     print(i)

        # # 3. Algorithm selection
        # if selection == "linearUCB":
        #     action, reward = linearUCB(possible_calendars, preference)

        # total_reward += reward
        # print("\ntotal_reward=", total_reward)

        # 3. Algorithm chooses which action to run
        # For simulation purpose, let's randomly choose one for now.
        # Algorithm: input: human_rating / human_correction from t-1. output: action_chosen at t.
        # print("============Actions Chosen============")
        # probability = [1/len(possible_calendars)] * len(possible_calendars)
        # action_index_chosen = np.random.choice(
        #     range(len(possible_calendars)), p=probability)
        # action_chosen = possible_calendars[action_index_chosen]
        # print("action_index_chosen=", action_index_chosen,
        #       ", action_chosen=", action_chosen)

        # # 4. Get reward
        # features_list = []
        # for each in relevant_events:
        #     for i in range(number_of_slots):
        #         # print("each, i: ", each, i)
        #         if action_chosen[i] == each:
        #             features_list.append(1)
        #         else:
        #             features_list.append(0)
        # print("============Get Reward============")
        # print("features_list=", features_list)
        # # Reward is for evaluating the algorithm
        # reward = np.dot(preference, features_list)
        # print("reward=", reward)
        # total_reward += reward

        # 5. Human feedback I
        # Getting human feedback from likert scale or thumbs up/down
        # For now, let's assume the human feedback is the same as the reward with some noise
        # TODO: Add algorithm to learn from human feedback here (estimate preference)
        # Paper: A Contextual-Bandit Approach to Personalized News Article Recommendation
        # noise = np.random.normal(loc=0.0, scale=0.1)
        # human_rating = reward + noise
        # TODO: change to integer (likert scale)
        # print("(Noisy) human_rating=", human_rating)

        # 6. Human feedback II
        # Getting human feedback from human correction
        # TODO: Add algorithm to learn from human feedback here (estimate preference)
        # Paper: Coactive Learning
        # calendar_highest_reward = None
        # reward_highest = -np.Inf
        # for eligible_calendar in possible_calendars:
        #     # update the feature list because the calendar has been updated
        #     updated_feature_list = featureListGenerator(
        #         eligible_calendar, [number_of_slots])
        #     eligible_calendar_reward = np.dot(preference, updated_feature_list)
        #     print("eligible_calendar=", eligible_calendar, ", features_list=",
        #           updated_feature_list, ", reward=", eligible_calendar_reward)

        #     # find the calendar with the highest reward
        #     if eligible_calendar_reward > reward_highest:
        #         reward_highest = eligible_calendar_reward
        #         calendar_highest_reward = eligible_calendar

        # print("calendar_highest_reward=", calendar_highest_reward)

        # assume the human correction is the optimal calendar
        # human_correction = calendar_highest_reward
        # print("(Noieless for now) human_correction=", human_correction)

        # Evaluation of the algorithm by the total reward
        # The higher total reward, the better
        # print("\n\ntotal_reward=", total_reward)

        # def testCase1():
        #     calendar1 = {1: "gym", 2: "assignment_out",
        #                  3: "reading", 4: -1, 5: "class"}
        #     simulation(calendar1)


def main():
    # TODO: offline procedure to extract information
    # testCase1()

    simulation(int(1e2))


if __name__ == "__main__":
    main()
