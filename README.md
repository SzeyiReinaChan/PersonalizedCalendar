# cs5100_spring24_project

Topic: AI-Assisted Personalized Calendars for Students
Team member: Kaiqi Zhang, Szeyi Chan

Problem formulation:<br>
1. A fixed set of relevant events 𝐸_𝑟, irrelevant events 𝐸_𝑖𝑟.<br>
2. A calendar 𝐶=K slots, e.g., 𝐶=[∅,∅,𝑒_1,𝑒_2,𝑒_1,∅].<br>
3. An action 𝑎 is to add the new events related to the syllabus.<br>
4. A state 𝑠 is the context (the irrelevant calendar).<br>
5. A reward 𝑟 maps [a calendar with irrelevant and relevant events allocated] to [a scalar].
3. For each episode t=1,…,T. The system observes the current state 𝑠_𝑡 and proposes an action 𝑎_𝑡.<br>
4. The action incurs reward 𝑟(s_𝑡,𝑎_𝑡), where the reward maps the calendar to a scalar.<br>
  4.1 No state transition, simplifying MDP to the contextual bandit, i.e., the system's decisions in previous rounds do not affect the user preference, the context, or the user feedback in future rounds.<br>

Assumption: <br>
1. Reward is linearly parameterized by a human preference vector θ and a binary feature vector describing the calendar[1].<br>
2. User preference is not changing across rounds.

Project question:<br>
Decision-making with the goal of max 𝑟 without knowing 𝑟 a-priori by requiring the system to learn about r using human feedback.<br>
In this project, we will simulate user preference and context(e.g., the irrelevant calendar which included events in the original calendar). We will leverage these modeled preferences to generate simulated user feedback. By analyzing feedback from student users, the system can organize syllabus events into student users' calendars more effectively.<br>
Goal: Compare the reward from 2 types of human feedback: (1) direct reward feedback[2]; (2) corrective feedback[3]<br>


[1] Gervasio, Melinda T., et al. "Active preference learning for personalized calendar scheduling assistance." Proceedings of the 10th international conference on Intelligent user interfaces. 2005.<br>
[2] Li, Lihong, et al. "A contextual-bandit approach to personalized news article recommendation." Proceedings of the 19th international conference on World wide web. 2010.<br>
[3] Shivaswamy, Pannaga, and Thorsten Joachims. "Coactive learning." Journal of Artificial Intelligence Research 53 (2015): 1-40.<br>
