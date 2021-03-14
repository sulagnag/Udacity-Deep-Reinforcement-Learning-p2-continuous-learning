# Udacity-Deep-Reinforcement-Learning-p2-continuous-learning
DDPG implementation for continuous action space

## The Problem Description

**The Environment:** In this environment, a double-jointed arm can move to target locations. The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). 

**Note:** The Unity ML-Agent team frequently releases updated versions of their environment. This project uses v0.4 interface. The project environment provided by Udacity is similar to, but not identical to the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment on the Unity ML-Agents GitHub page.

For this project, Udacity provides two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

**The observation space:** The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

**The action space:** Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**Rewards:**  A reward of +0.1 is provided for each step that the agent's hand is in the goal location

**Task (Episodic/Continuous)**: The task is continuos, so we use a limit of max timesteps for each episode.

**Solution:**  I have decided to go with the second version. 
**The agents thus need to be trained so they get an average score of +30 (over 100 consecutive episodes, and over all agents)**


## Getting started

### Installation requirements

- To begin with, you need to configure a Python 3.6 / PyTorch 0.4.0 environment with the requirements described in [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
- Then you need to clone this project and have it accessible in your Python environment
- For this project, you will not need to install Unity. You need to only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
- Finally, you can unzip the environment archive in the project's environment directory and set the path to the UnityEnvironment in the code.

## Instructions
The configuration for the environement, the agent and the DDPG parameters are mentioned in the config file.

