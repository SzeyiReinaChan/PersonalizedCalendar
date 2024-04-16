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
# irrelevant_events = []
relevant_events = ["assignment_out", "reading"]
# relevant_events = ["reading"]
events = irrelevant_events + relevant_events

# events can only be placed in certain slots
assignment_out_constrain = [0, 1, 2]
reading_constrain = [0, 1, 2, 3, 4]

# Total number of slots (5 for now for simplicity)
# Will be changed to 24*60/30 = 48 for 30 minutes slots for day implementation
number_of_slots = 5

ucb_reward_dataset = []
ucb_regret_dataset = []

# For generating the preference for each event in each slot.


def preferenceGenerator(calendar):
    preference_count = 0
    for each in calendar.values():
        if each in relevant_events:
            preference_count += 1
    # print("preference_count=", preference_count)

    # randomly generated θ(preference) for evaluation, e.g., θ is uniformly random
    preference = np.random.uniform(0, 1, preference_count*number_of_slots)

    return preference


def featureListGenerator(input, num_range):
    features_list = []
    for each in relevant_events:
        for i in range(*num_range):
            # #print("each, i: ", each, i)
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
            # print("each_position=", each_position)

            if st_irrelevant_calendar[each_position] == -1:
                st_irrelevant_calendar[each_position] = each
            else:
                # if there is already an event, skip
                # print("Skipped placement")
                continue
        # print("st_irrelevant_calendar=", st_irrelevant_calendar)
    return st_irrelevant_calendar


def eligibleActionsGenerator(st_irrelevant_calendar):
    slots_available = [i for i in range(
        number_of_slots) if st_irrelevant_calendar[i] == -1]
    # print("slots_available=", slots_available)

    possible_calendars = []

    # slot_for_assignment = copy.copy(slots_available)
    # for slot_index in range(len(slot_for_assignment)):
    #     slot_for_assignment.remove(slot_index)
    #     for each in relevant_events:
    #         constrain_name = each + "_constrain"
    #         if slot_index in globals()[constrain_name]:
    #             new_calendar = copy.copy(st_irrelevant_calendar)
    #             new_calendar[slot_index] = each
    #             possible_calendars.append(new_calendar)

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

    # for relevant_event_index in range(len(relevant_events)): # index: 0, 1
    #     for slot_index in range(len(slots_available)):  # index: 0, 1
    #         name = relevant_events[relevant_event_index]  # reading
    #         constrain_name = name + "_constrain"  # reading_constrain
    #         #print(constrain_name, "=", globals()[constrain_name])
    #         #print(slot_index)
    #         if slots_available[slot_index] != -1:
    #             continue

    # slot_for_assignment = copy.copy(slots_available)
    # for slot_index in range(len(slot_for_assignment)):
    #     slot_for_assignment.remove(slot_index)
    #     for each in relevant_events:
    #         constrain_name = each + "_constrain"
    #         if slot_index in globals()[constrain_name]:

    #     #print("slot_available=", slot_for_assignment)
    #     for relevant_event_index in range(len(relevant_events)):
    #         name = relevant_events[relevant_event_index]
    #         constrain_name = name + "_constrain"
    #         if slot_index in globals()[constrain_name]:
    #             new_calendar = copy.copy(st_irrelevant_calendar)
    #             new_calendar[slot_index] = name
    #             possible_calendars.append(new_calendar)

    # print("============Possible Actions============")
    # for i in possible_calendars:
        # print(i)
    return slots_available, possible_calendars


def plot_data(dataset_1, dataset_2, title, plot_filename):
    # reward = dataset_1
    regret = dataset_2
    # plt.plot(reward)
    plt.plot(regret)
    plt.xlabel('rounds')
    plt.ylabel('reward value')
    plt.title(title)
    plt.legend(['reward', 'regret'], loc='upper left')
    plt.savefig(plot_filename)
    plt.close()


def simulation(rounds=int(1e5)):
    # simulate preference
    # preference = preferenceGenerator(calendar)
    # preference = np.random.uniform(
    #     0, 1, len(relevant_events)*number_of_slots)
    preference = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    # print("============Preference============")
    # print("preference=", preference)

    # initialize the algorithm
    linear_ucb = LinearUCB(relevant_events, number_of_slots)
    # initialize reward
    ucb_total_reward = 0.0
    ucb_total_regret = 0.0

    # Rounds of simulations
    for t in range(rounds):
        # print("\n============ Round", t+1, "============")
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
        # print("ucb action chosen=", ucb_chosen_action_at)

        # get the reward for the best possible calendar
        # all_possible_reward = []
        # for each_feature in feature_factors:
        #     possible_reward = np.dot(preference, np.array(each_feature))
        #     all_possible_reward.append(possible_reward)
        # best_possible_reward = max(all_possible_reward)
        # #print("ucb best possible reward=", best_possible_reward)

        best_reward = -np.inf
        best_reward_feature = None
        for each_feature in feature_factors:
            possible_reward = np.dot(preference, np.array(each_feature))
            if possible_reward > best_reward:
                best_reward = possible_reward
                best_reward_feature = each_feature
        best_possible_reward = best_reward
        # print("best_reward=", best_reward)
        # print("best_reward_feature=", best_reward_feature)

        # get the reward - reward of the true human preference
        chosen_action_feature_list = featureListGenerator(
            ucb_chosen_action_at, [number_of_slots])
        ucb_human_reward = np.dot(preference, chosen_action_feature_list)
        # print("ucb human reward=", ucb_human_reward)

        # update the reward based on the human rating + random simulated noise
        ucb_noisy_reward = ucb_human_reward + \
            np.random.normal(loc=0.0, scale=0.1)
        # ucb_noisy_reward = ucb_human_reward
        # print("ucb noisy reward=", ucb_noisy_reward)

        # getting the regression number: how far away from the best possible reward?
        regret_number = best_possible_reward-ucb_noisy_reward
        # print("ucb regression_number=", regret_number)

        # update the algorithm based on the reward
        linear_ucb.updateByRating(
            ucb_noisy_reward, possible_calendars, feature_factors)

        # for evaluation purpose
        ucb_total_reward += ucb_noisy_reward
        # print("\nucb total_reward=", ucb_total_reward)
        ucb_total_regret += regret_number
        # print("ucb total_regret=", ucb_total_regret)

        # store the reward for plotting
        ucb_reward_dataset.append(ucb_total_reward)
        ucb_regret_dataset.append(ucb_total_regret)

    plot_data(ucb_reward_dataset, ucb_regret_dataset, "UCB", "ucb.png")


def main():
    # TODO: offline procedure to extract information

    simulation()


if __name__ == "__main__":
    main()
