This repo contains solution for Reacher Unity Environment as part of udacity Deep Reinforced Learning Nanodegree program.

Problem:
Make a agent learn to move a double jointed arm at a target position. This is the second version of environment where 20 arms are connected to a single brain.

For more details please check out this link: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher

Actions:
this environment has a continuous action space of size 4.

Rewards: +0.1 Each step agent's hand is in goal location.

To solve this environment, we need to acheive a average score of 30 or above.

Dependencies File:
	requirements.txt
	use command : pip install -r requirements.txt

Main File: Reacher_Main.ipynb
Running Trained Agent File: Reacher_Trained.py
Saved Weights File: checkpoint_actor.pth , checkpoint_critic.pth
