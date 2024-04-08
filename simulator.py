import numpy as np
import random
import copy
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


def preference_generator(calendar):
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


def simulation(rounds=2):
    # initialize_feature = featureListGenerator(
    #     calendar, [1, number_of_slots + 1])
    # print("============**Init**============")
    # print("initialize_feature= ", initialize_feature)

    # simulate preference
    # preference = preference_generator(calendar)
    preference = np.random.uniform(
        0, 1, len(relevant_events)*number_of_slots)
    print("============Preference============")
    print("preference=", preference)

    # assert (len(preference) == len(initialize_feature))

    # initialize reward
    total_reward = 0.0
    # # 10 rounds of simulation
    for t in range(rounds):
        print("\n============ Round ", t, "============")

        # 1. Get st=irrelevant calendar
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
                    continue
            print("st_irrelevant_calendar=", st_irrelevant_calendar)

        # 2. Find all eligible actions
        # E.g., constraint: in this week, both assignment_out and reading have to be assigned.
        # assignment_out has to be slot 0,1,2. reading can be anytime.
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

        # 3. Algorithm chooses which action to run
        # For simulation purpose, let's randomly choose one for now.
        # Algorithm: input: human_rating / human_correction from t-1. output: action_chosen at t.
        print("============Actions Chosen============")
        probability = [1/len(possible_calendars)] * len(possible_calendars)
        action_index_chosen = np.random.choice(
            range(len(possible_calendars)), p=probability)
        action_chosen = possible_calendars[action_index_chosen]
        print("action_index_chosen=", action_index_chosen,
              ", action_chosen=", action_chosen)

        # 4. Get reward
        features_list = []
        for each in relevant_events:
            for i in range(number_of_slots):
                # print("each, i: ", each, i)
                if action_chosen[i] == each:
                    features_list.append(1)
                else:
                    features_list.append(0)
        print("============Get Reward============")
        print("features_list=", features_list)
        # Reward is for evaluating the algorithm
        reward = np.dot(preference, features_list)
        print("reward=", reward)
        total_reward += reward

        # 5. Human feedback I
        # Getting human feedback from likert scale or thumbs up/down
        # For now, let's assume the human feedback is the same as the reward with some noise
        # TODO: Add algorithm to learn from human feedback here (estimate preference)
        # Paper: A Contextual-Bandit Approach to Personalized News Article Recommendation
        noise = np.random.normal(loc=0.0, scale=0.1)
        human_rating = reward + noise
        # TODO: change to integer (likert scale)
        print("(Noisy) human_rating=", human_rating)

        # 6. Human feedback II
        # Getting human feedback from human correction
        # TODO: Add algorithm to learn from human feedback here (estimate preference)
        # Paper: Coactive Learning
        calendar_highest_reward = None
        reward_highest = -np.Inf
        for eligible_calendar in possible_calendars:
            # update the feature list because the calendar has been updated
            updated_feature_list = featureListGenerator(
                eligible_calendar, [number_of_slots])
            eligible_calendar_reward = np.dot(preference, updated_feature_list)
            print("eligible_calendar=", eligible_calendar, ", features_list=",
                  updated_feature_list, ", reward=", eligible_calendar_reward)

            # find the calendar with the highest reward
            if eligible_calendar_reward > reward_highest:
                reward_highest = eligible_calendar_reward
                calendar_highest_reward = eligible_calendar

        # print("calendar_highest_reward=", calendar_highest_reward)

        # assume the human correction is the optimal calendar
        human_correction = calendar_highest_reward
        print("(Noieless for now) human_correction=", human_correction)

    # Evaluation of the algorithm by the total reward
    # The higher total reward, the better
    print("\n\ntotal_reward=", total_reward)


# def testCase1():
#     calendar1 = {1: "gym", 2: "assignment_out",
#                  3: "reading", 4: -1, 5: "class"}
#     simulation(calendar1)

def main():
    # TODO: offline procedure to extract information
    # testCase1()

    simulation()


if __name__ == "__main__":
    main()
