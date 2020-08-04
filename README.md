# Reacher
Solution to the second project of Udacitys Reinforcement Learning Nanodegree

![video](https://user-images.githubusercontent.com/63595824/89272032-db5ee580-d63d-11ea-8cbc-0d39483c4b31.gif)

### Quick Installation

To set up the python environment and install all requirements for using this repository on Linux OS, follow the instructions given below:
1. Create and activate a new environment with Python 3:
    ```bash
    python3 -m venv /path/to/virtual/environment
    source /path/to/virtual/environment/bin/activate
    ```
2. Clone the Udacity repository and navigate to the `python/` folder to install the `unityagents`-package:
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    cd -
    ```
3. Download the Unity-environment from Udacity and unzip it:
    ```bash
    wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
    unzip Reacher_Linux.zip
    rm Reacher_Linux.zip
    ```

For using this repository using Windows OS or Mac OSX, please follow the instructions given ![here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Afterwards download the Unity-environment for ![Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip), ![Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) or ![Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) and unzip it in this repository. If run under one of these operating systems, the path in the class ```Env```  in ```envwrapper.py``` has to be adjusted in such a way, that it points to the corresponding executable.



### Quick Start

After installing all requirements and activating the virtual environment training the agent can be started by executing

```bash
python main.py
```

Configurations like enabling the visualization or adjustments to the architecture is possible through the dictionaries defined in `main.py`.
During training a file called `performance.log` is created, which holds information about the current score and the average score of the last 100 episodes. Furthermore, if the agents variable `save_after` is set to a value larger than 0, after the given number of epochs the agents current parameters will be saved in a file with the agents name plus `_parameters.dat`, while the current weights of the agents models will be saved in a file with the agents name plus `_actor.model` and `_actor_target.model` as well as `_critic.model` and `_critic_target.model`.

Running

```bash
python evaluate.py
```

allows to evaluate the performance of the saved agent of the given name. 


### Background Information
In this environment the player (or agent) has to control double-jointed arms in such a way, that they follow a given goal location. The aim is to maintain the arms position at the target location for as long as possible. In every timestep in which the agent manages to do so, he gets a reward of **`+0.04`**. At every timestep, the player is provided a 33-dimensional state (with information about position, rotation, velocity and angular velocity of the arm) per arm and has to decide on the actions to take. For every arm, the corresponding action-vector consists of 4 real numbers in the range from -1 to +1. After a given time (1000 timesteps) the game is over, hence making this task an episodic one. The environment is considered solved when the player achieves an average score of **`+30`** over 100 consecutive episodes, where the score is given as the average of the scores achieved by all arms in a single iteration.

For more information on the approach that was used to solve this environment, see [`Report.md`](https://github.com/fberressem/Reacher/blob/master/Report.md).

