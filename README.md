The Minesweeper AI bot project is designed to automate gameplay in the classic Minesweeper game using advanced reinforcement learning techniques, specifically Deep Q-Networks (DQN) and Dueling Deep Q-Networks (DDQN). The project is implemented in Python and utilizes the PyTorch library for building and training neural network models. 

The core of the project consists of a dynamic game environment, and methods for interacting with the game. The Minesweeper game environment handles game state initialization, updates, and resets. The project contains two main model classes. Each class encapsulates a model (DQN or DDQN), The game's graphical interface provides a visual representation of the game state if enabled. 

Overall, this project demonstrates the application of modern reinforcement learning techniques to a traditional puzzle game, showcasing the potential of AI to solve complex problems in a structured environment. 

 
 
 

Working of each python file in the repo:

env.py : environment
display.py : pygame renderer file
main.py : win tester and slow 5 games demo codes for DQN and DDQN

DQNmodel.py and DDQNmodel.py : NN models
trainDQNmodel.py and trainDDQNmodel.py : training codes

DQN_plotter.py and DDQN_plotter.py : plotting files

trained_checkpoints_DQN and trained_checkpoints_DDQN : saved checkpoints at every 1000 epochs while training models
training_logs_DQN and training_logs_DDQN : training data (win rate, loss, avg rewards and epsilon decay values)