import torch
from DDQNmodel import DuelingDQN
from DQNmodel import DQN
from env import MineSweeper
from display import DisplayGame
import time


def main():
    # playMultipleGames_DQN(1000, True)
    # play5games_DQN(5,True)
    
    playMultipleGames_DDQN(1000, True)
    # play5games_DDQN(5,True)


class AI_Player_DQN():
    def __init__(self, render_flag):
        self.model = DQN(36, 36)  
        self.render_flag = render_flag  # For GUI visualization on/off
        self.width = 6  # To Change For Maze width
        self.height = 6  # To Change For Maze height
        self.bombs = 6  # To Change For Number of mines 
        self.env = MineSweeper(self.width, self.height, self.bombs)
        if self.render_flag:
            self.renderer = DisplayGame(self.env.state)
        model_number = 20000  # To Change For Different Models
        self.load_models(model_number)

    def get_action(self, state):
        state = state.flatten()
        mask = (1 - self.env.fog).flatten()
        action = self.model.act(state, mask)
        return action

    def load_models(self, number):
        path = "trained_checkpoints_DQN/trained_for_" + str(number) + ".pth"
        dictTemp = torch.load(path)
        self.model.load_state_dict(dictTemp['current_state_dict'])
        self.model.epsilon = 0

    def do_step(self, action):
        i = int(action / self.width)
        j = action % self.width
        if self.render_flag:
            self.renderer.state = self.env.state
            self.renderer.draw()
            self.renderer.bugfix()
        next_state, terminal, reward = self.env.choose(i, j)
        return next_state, terminal, reward


def playMultipleGames_DQN(games_no, display_games=True):
    tester = AI_Player_DQN(display_games)
    state = tester.env.state
    mask = tester.env.fog
    wins = 0
    i = 0
    step = 0
    first_loss = 0
    while i < games_no:
        step += 1
        action = tester.get_action(state)
        next_state, terminal, reward = tester.do_step(action)
        state = next_state
        if terminal:
            if step == 1 and reward == -1:
                first_loss += 1
            i += 1
            tester.env.reset()
            state = tester.env.state
            if reward == 1:
                wins += 1
            step = 0

    # print("Trained using DQN : Win Rate For " + str(games_no) + " Games: " + str(wins * 100 / (games_no)))
    print("Win Rate using DQN: " + str(wins * 100 / (games_no - first_loss)))


def play5games_DQN(games_no=5, display_games=True):
    
    tester = AI_Player_DQN(True)
    state = tester.env.state
    count = 0
    start = time.perf_counter()
    step = 0
    first_loss = 0
    i=0


    while(i<games_no):
        count+=1
        step+=1
        action = tester.get_action(state)
        next_state,terminal,reward = tester.do_step(action)
        state = next_state
        print(reward)
        time.sleep(0.5)

        if(terminal):
            if(reward==1):
                print("WIN")
            else:
                
                print("LOSS")
            i+=1
            tester.env.reset()
            step=0
            state = tester.env.state
            # break

class AI_Player_DDQN():
    def __init__(self, render_flag):
        self.model = DuelingDQN(36, 36)
        self.render_flag = render_flag  # For GUI visualization on/off
        self.width = 6  # To Change For Maze width
        self.height = 6  # To Change For Maze height
        self.bombs = 6  # To Change For Number of mines 
        self.env = MineSweeper(self.width, self.height, self.bombs)
        if self.render_flag:
            self.renderer = DisplayGame(self.env.state)
        model_number = 19000  # To Change For Different Models
        self.load_models(model_number)

    def get_action(self, state):
        state = state.flatten()
        mask = (1 - self.env.fog).flatten()
        action = self.model.act(state, mask)
        return action

    def load_models(self, number):
        path = "trained_checkpoints_DDQN/trained_for_" + str(number) + ".pth"
        dictTemp = torch.load(path)
        self.model.load_state_dict(dictTemp['current_state_dict'])
        self.model.epsilon = 0

    def do_step(self, action):
        i = int(action / self.width)
        j = action % self.width
        if self.render_flag:
            self.renderer.state = self.env.state
            self.renderer.draw()
            self.renderer.bugfix()
        next_state, terminal, reward = self.env.choose(i, j)
        return next_state, terminal, reward


def playMultipleGames_DDQN(games_no, display_games=True):
    tester = AI_Player_DDQN(display_games)
    state = tester.env.state
    mask = tester.env.fog
    wins = 0
    i = 0
    step = 0
    first_loss = 0
    while i < games_no:
        step += 1
        action = tester.get_action(state)
        next_state, terminal, reward = tester.do_step(action)
        state = next_state
        if terminal:
            if step == 1 and reward == -1:
                first_loss += 1
            i += 1
            tester.env.reset()
            state = tester.env.state
            if reward == 1:
                wins += 1
            step = 0

    # print("Trained using Dueling DQN :Win Rate For " + str(games_no) + " Games: " + str(wins * 100 / (games_no)))
    print("Win Rate using Dueling DQN: " + str(wins * 100 / (games_no - first_loss)))


def play5games_DDQN(games_no=5, display_games=True):
    
    tester = AI_Player_DDQN(True)
    state = tester.env.state
    count = 0
    start = time.perf_counter()
    step = 0
    first_loss = 0
    i=0

    while(i<games_no):
        count+=1
        step+=1
        action = tester.get_action(state)
        next_state,terminal,reward = tester.do_step(action)
        state = next_state
        print(reward)
        time.sleep(0.5)

        if(terminal):
            if(reward==1):
                print("WIN")
            else:
                
                print("LOSS")
            i+=1
            tester.env.reset()
            step=0
            state = tester.env.state
            # break
            
main()
