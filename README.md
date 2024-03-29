# cs5100_spring24_project
Project question:
Decision-making with the goal of max 𝑟 without knowing 𝑟 a-priori by requiring S to learn about r using human feedback.
-compare 2 types of human feedback: (1) direct reward feedback[1]; (2) corrective feedback[2]

Assumption: reward is linearly parameterized by a human preference vector θ and a binary feature vector describing the calendar[3].

Problem formulation:
1. A fixed set of relevant events 𝐸_𝑟, irrelevant events 𝐸_𝑖𝑟.
2. A calendar 𝐶=K slots, e.g., 𝐶=[∅,∅,𝑒_1,𝑒_2,𝑒_1,∅].
3. For each episode t=1,…,T. S observes the current state 𝑠_𝑡 and proposes an action 𝑎_𝑡.
4. The action incurs reward 𝑟(s_𝑡,𝑎_𝑡 ). (No state transition, simplifying MDP to contextual bandit.)

[1] Li, Lihong, et al. "A contextual-bandit approach to personalized news article recommendation." Proceedings of the 19th international conference on World wide web. 2010.
[2] Shivaswamy, Pannaga, and Thorsten Joachims. "Coactive learning." Journal of Artificial Intelligence Research 53 (2015): 1-40.
[3] Gervasio, Melinda T., et al. "Active preference learning for personalized calendar scheduling assistance." Proceedings of the 10th international conference on Intelligent user interfaces. 2005.