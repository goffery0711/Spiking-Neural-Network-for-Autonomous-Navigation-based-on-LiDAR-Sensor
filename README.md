# Spiking Neural Network for Autonomous Navigation based on LiDAR Sensor

This repository contains the code of my Master's thesis "[Spiking Neural Network for Autonomous Navigation based on LiDAR Sensor](RAIStudentThesis_ZhenZhou.pdf)":

- **Controller:** Contains the controller code as well as Matplotlib plots.
- **V-REP Scenarios:** The V-REP scene files for 4 different lane following scenarios as well as the Lua script handling the communication between robot and controller via ROS can be found here.
- **RAIStudentThesis_ZhenZhou.pdf:** Thesis PDF file

## Abstract

Deep Q-learning algorithms in reinforcement learning have been used in vehicle training, and this technology is relatively mature. However, artificial neural network (ANN) can suffer from long response times, high energy consumption, and high training costs due to their own connection methods. A new solution has been found in event-based spiking neural networks(SNN) that imitate biological neurons. In Reinforcement Learning tasks, SNNs can be trained to better control the direction of robot movement. This explores new ideas for the development of future autonomous driving technologies. In this thesis, a simulated lane tracking task is implemented using LiDAR. Firstly, the DQN algorithm and the Reward-modulated Spiking-Timing-Dependent-Plasticity (R-STDP) are compared by training on the same task. After coming up with the better R-STDP algorithm, the whole algorithm is optimized by different environment perception methods. What performs better among these is by combining Fully Convolutional Network (FCN) to predict the driving area and then obtain the lane edges by edge extraction as input state. Finally, the robustness of the algorithm is tested in scenarios of different complexity. Although the algorithm currently lacks the required reward prediction capability for more complex decision tasks, there are quite a few improvements in the adapt-ability to the environment and the response speed. Future optimization of this algorithm can be based on this algorithm to deal with more complex tasks in real-world situations.

## Start

1. Start ROS node ("roscore").
2. Start CoppeliaSim(V-REP) with activated RosInterface (http://www.coppeliarobotics.com/helpFiles/en/rosTutorialIndigo.htm).
3. Load lane following scenario.
4. Start Simulation

## Controller

#### DQN-SNN

1. **Parameters:** Before training all important parameters can be set in the *parameters.py* file. 
Most importantly, for each session a subfolder has to be created and named in the *path* variable where all the data is stored.
2. **DQN training:** Execute *dqn_training.py* to start the DQN training process. During training, the episode number and number of total steps are displayed
in the console. In order to show rewards and episode length during training, start "tensorboard --logdir=\<path\>". 
When training is finished (maximum total steps reached), network weights as well as state-action dataset is stored in the *dqn_data.h5* file.
3. **DQN controller:** After training, the DQN controller can be tested executing the *dqn_controller.py* file. 
The controller performs one lap on the outer lane of the course using the stored data. 
During the lap, the robot's position on the course, as well as its distance to the lane-center are recorded and stored afterwards in the *dqn_performance_data.h5* file.
4. **SNN training:** For transferrng the previously learned DQN policy to a SNN, the *snn_training.py* executable uses the state-action dataset
stored in the *dqn_data.h5* file and trains another tensorflow neural network on it. Learned weights are stored in the *snn_data.h5* file afterwards.
5. **SNN controller:** The weights stored in *snn_data.h5* can be used by the SNN controller *snn_controller.py*. 
It performs one lap on the outer lane of the course and stores robot position and lane-center distance as well in *snn_performance_data.h5*.

#### Braitenberg

1. **Parameters:** Before training all important parameters can be set in the *parameters.py* file. 
Most importantly, for each session a subfolder has to be created and named in the *path* variable where all the data is stored.
2. **Braitenberg controller:** The controller can be started by executing *controller.py*. It performs a single lap on the outer lane 
of the course and stores robot position as well as lane-center distance to *braitenberg_performance_data.h5*.

#### R-STDP

1. **Parameters:** Before training all important parameters can be set in the *parameters.py* file. 
Most importantly, for each session a subfolder has to be created and named in the *path* variable where all the data is stored.
2. **R-STDP training:** Execute *training.py* for training the R-STDP synapses. The training length can be adjusted in the *parameters.py* file.
 During training, network weights and termination positions are stored and saved in the *rstdp_data.h5* file afterwards.
3. **R-STDP controller:** After training, the R-STDP controller can be tested executing the *controller.py* file. 
The controller performs one lap on the outer lane of the course using the previously learned weights. 
During the lap, the robot's position on the course, as well as its distance to the lane-center are recorded and stored afterwards in the *rstdp_performance_data.h5* file.

## Software versions:

 - Ubuntu 18.04
 - CoppeliaSim 4.1.0
 - ROS Melodic 1.14.10
 - Python 2.7
 - Tensorflow 1.15.0
 - NEST 2.14.0

