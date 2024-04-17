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

    # print("============Possible Actions============")
    # for i in possible_calendars:
        # print(i)
    return slots_available, possible_calendars


def calculation_and_plotting(rep, rounds, reward_dataset, regret_dataset,
                             reward_over_t_dataset, regret_over_t_dataset, algo_name):
    reward_avg = np.mean(reward_dataset, axis=0)
    regret_avg = np.mean(regret_dataset, axis=0)

    # calculate the standard error for plotting error bar
    reward_std_err = np.std(reward_dataset, axis=0) / np.sqrt(rep)
    regret_std_err = np.std(regret_dataset, axis=0) / np.sqrt(rep)

    # for decay plot
    reward_over_t_avg = np.mean(reward_over_t_dataset, axis=0)
    regret_over_t_avg = np.mean(regret_over_t_dataset, axis=0)

    reward_over_t_std_err = np.std(
        reward_over_t_dataset, axis=0) / np.sqrt(rep)
    regret_over_t_std_err = np.std(
        regret_over_t_dataset, axis=0) / np.sqrt(rep)
    # print(regret_over_t_dataset)
    # print(regret_over_t_std_err)

    # plot the data
    # input: (dataset, dataset_name, xlabel, ylabel, title, plot_filename)
    plot_data(regret_avg, algo_name + " regret set",
              "rounds", "value", algo_name, algo_name+".png")
    # decay plot suppose to show that with more rounds, the regret will decrease
    plot_data(regret_over_t_avg, algo_name + " regret set",
              "rounds", "value", algo_name, algo_name+"_decay.png")

    # input: (rounds, algo_name, dataset, std_err_dataset, xlabel, ylabel, title, plot_filename)
    plot_data_with_error_bar(rounds, algo_name, regret_avg, regret_std_err,
                             "rounds", "value", algo_name, algo_name+"_with_error_bar.png")
    plot_data_with_error_bar(rounds, algo_name, regret_over_t_avg, regret_over_t_std_err,
                             "rounds", "value", algo_name, algo_name+"decay_with_error_bar.png")
    return reward_avg, regret_avg, reward_std_err, regret_std_err, reward_over_t_avg, regret_over_t_avg, reward_over_t_std_err, regret_over_t_std_err


def plot_data(dataset, dataset_name, xlabel, ylabel, title, plot_filename):
    plt.plot(dataset)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([dataset_name], loc='upper left')
    plt.savefig(plot_filename)
    plt.close()

def plot_2_data_sets(dataset_1, dataset_2, dataset_1_legend, dataset_2_legend, title, plot_filename):
    plt.plot(dataset_1)
    plt.plot(dataset_2) 
    plt.xlabel('rounds')
    plt.ylabel('value')
    plt.title(title)
    plt.legend([dataset_1_legend, dataset_2_legend], loc='upper left')
    plt.savefig(plot_filename)
    plt.close()

def plot_data_with_error_bar(rounds, algo_name, dataset, std_err_dataset, xlabel, ylabel, title, plot_filename):
    time = np.arange(rounds)
    plt.plot(time, dataset, label='Average ' + algo_name + ' Regret')
    plt.fill_between(time,
                     dataset - std_err_dataset,
                     dataset + std_err_dataset,
                     color='lightblue', alpha=0.2, label='Standard Error')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(plot_filename)
    plt.close()


def simulation(rounds=int(2500)):
    ucb_reward_dataset = []
    ucb_regret_dataset = []
    ucb_reward_over_t = []
    ucb_regret_over_t = []

    pp_reward_dataset = []
    pp_regret_dataset = []
    pp_reward_over_t = []
    pp_regret_over_t = []

    # simulate preference
    preference = np.random.uniform(
        0, 1, len(relevant_events)*number_of_slots)
    # print("============Preference============")
    # print("preference=", preference)

    # initialize the algorithm
    linear_ucb = LinearUCB(relevant_events, number_of_slots)
    # initialize reward
    ucb_total_reward = 0.0
    ucb_total_regret = 0.0

    pp_total_reward = 0.0
    pp_total_feedback_reward = 0.0
    pp_total_regret = 0.0

    #initialize weights for perceptron
    num_features = len(relevant_events)*number_of_slots
    w = np.zeros(num_features)

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
        ucb_reward_over_t.append(ucb_total_reward/float(t+1))
        ucb_regret_over_t.append(ucb_total_regret/float(t+1))

        # Preference Perceptron simulation===================================
        best_action = None
        max_val = float('-inf')
        for calendar in possible_calendars:
            feature_vector = featureListGenerator(calendar, [number_of_slots])
            val = np.dot(w.T, feature_vector)
            if val > max_val:
                max_val = val
                best_action = calendar
        #Get reward for best action based on preference
        pp_reward = np.dot(preference, featureListGenerator(best_action, [number_of_slots]))
        # print("Reward for best action based on Perceptron: ", pp_reward)
        pp_total_reward += pp_reward

        #obtain feedback by selecting from calendars with rewards greater than or equal to the best action reward
        feedback_calendars = []
        for calendar in possible_calendars:
            calendar_reward = np.dot(preference, featureListGenerator(calendar, [number_of_slots]))
            if calendar_reward >= pp_reward:
                feedback_calendars.append(calendar)
        if not feedback_calendars:
            feedback_calendars.append(best_action)

        #Randomly select a feedback calendar
        feedback_index = np.random.choice(len(feedback_calendars))
        feedback = feedback_calendars[feedback_index]
        feedback_reward = np.dot(preference, featureListGenerator(feedback, [number_of_slots]))
        pp_total_feedback_reward += feedback_reward

        #Regret calculation
        best_possible_reward_pp = max(np.dot(preference, featureListGenerator(calendar, [number_of_slots])) for calendar in possible_calendars)
        pp_total_regret += best_possible_reward_pp - pp_reward
        
        phi_best_action = np.array(featureListGenerator(best_action, [number_of_slots]))
        phi_feedback = np.array(featureListGenerator(feedback, [number_of_slots]))

        #Update weights
        w +=  phi_feedback - phi_best_action 

        pp_reward_dataset.append(pp_total_reward)
        pp_regret_dataset.append(pp_total_regret)
        pp_reward_over_t.append(pp_total_reward / (t + 1))
        pp_regret_over_t.append(pp_total_regret / (t + 1))

    return rounds, ucb_reward_dataset, ucb_regret_dataset, ucb_reward_over_t, ucb_regret_over_t, pp_reward_dataset, pp_regret_dataset, pp_reward_over_t, pp_regret_over_t


def main():
    # run 10 times simulation, each time 100 rounds
    ucb_reward_dataset = None
    ucb_regret_dataset = None
    ucb_reward_over_t_dataset = None
    ucb_regret_over_t_dataset = None

    pp_reward_dataset = None
    pp_regret_dataset = None
    pp_reward_over_t_dataset = None
    pp_regret_over_t_dataset = None

    rep = 10
    for _ in range(rep):
        rounds, ucb_reward_data, ucb_regret_data, ucb_reward_over_t_data, ucb_regret_over_t_data,\
        pp_reward_data, pp_regret_data, pp_reward_over_t_data, pp_regret_over_t_data = simulation()
        if ucb_reward_dataset is None:
            ucb_reward_dataset = np.array([ucb_reward_data])
            ucb_regret_dataset = np.array([ucb_regret_data])
            ucb_reward_over_t_dataset = np.array([ucb_reward_over_t_data])
            ucb_regret_over_t_dataset = np.array([ucb_regret_over_t_data])
        else:
            ucb_reward_dataset = np.vstack((ucb_reward_dataset, ucb_reward_data))
            ucb_regret_dataset = np.vstack((ucb_regret_dataset, ucb_regret_data))
            ucb_reward_over_t_dataset = np.vstack((
                ucb_reward_over_t_dataset,ucb_reward_over_t_data))
            ucb_regret_over_t_dataset = np.vstack((ucb_regret_over_t_dataset, ucb_regret_over_t_data))
        
        # Store PP results
        if pp_reward_dataset is None:
            pp_reward_dataset = np.array([pp_reward_data])
            pp_regret_dataset = np.array([pp_regret_data])
            pp_reward_over_t_dataset = np.array([pp_reward_over_t_data])
            pp_regret_over_t_dataset = np.array([pp_regret_over_t_data])
        else:
            pp_reward_dataset = np.vstack((pp_reward_dataset, pp_reward_data))
            pp_regret_dataset = np.vstack((pp_regret_dataset, pp_regret_data))
            pp_reward_over_t_dataset = np.vstack((pp_reward_over_t_dataset, pp_reward_over_t_data))
            pp_regret_over_t_dataset = np.vstack((pp_regret_over_t_dataset, pp_regret_over_t_data))

    reward_avg_ucb, regret_avg_ucb, reward_std_err_ucb, regret_std_err_ucb, reward_over_t_avg_ucb, regret_over_t_avg_ucb,\
    reward_over_t_std_err_ucb, regret_over_t_std_err_ucb = calculation_and_plotting(rep, rounds, ucb_reward_dataset, 
                                                                                    ucb_regret_dataset, ucb_reward_over_t_dataset, 
                                                                                    ucb_regret_over_t_dataset, 'ucb')
    reward_avg_pp, regret_avg_pp, reward_std_err_pp, regret_std_err_pp, reward_over_t_avg_pp, regret_over_t_avg_pp,\
    reward_over_t_std_err_pp, regret_over_t_std_err_pp = calculation_and_plotting(rep, rounds, pp_reward_dataset, 
                                                                                pp_regret_dataset, pp_reward_over_t_dataset, 
                                                                                pp_regret_over_t_dataset, 'pp')
    plot_2_data_sets(regret_avg_ucb, regret_avg_pp, 'UCB', 'PP', 'Regret Comparison', 'regret_comparison.png')
    plot_2_data_sets(regret_over_t_avg_ucb, regret_over_t_avg_pp, 'UCB', 'PP', 'Regret Decay Comparison', 'regret_over_t_comparison.png')


if __name__ == "__main__":
    main()
