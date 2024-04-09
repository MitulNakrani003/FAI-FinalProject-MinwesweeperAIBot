import torch
from model import DuelingDeepQNetwork
from env import MineSweeper
from display import DisplayGame
import time


def main():
    PlayMultipleGames(20000, False)
    PlayAGame(True)


class AI_Player():
    def __init__(self, render_flag):
        self.model = DuelingDeepQNetwork(36, 36)  # To Change For Different Models
        self.render_flag = render_flag
        self.width = 6  # To Change For Maze Size
        self.height = 6  # To Change For Maze Size
        self.bombs = 6  # To Change For Bombs
        self.env = MineSweeper(self.width, self.height, bomb_no=self.bombs)
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
        path = "trained_checkpoints/trained_for_" + str(number) + ".pth"
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


def PlayMultipleGames(games_no, display_games=False):
    tester = AI_Player(display_games)
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

    print("Trained Win Rate For " + games_no + " Games: " + str(wins * 100 / (games_no)))
    print("Win Rate Excluding First Loss: " + str(wins * 100 / (games_no - first_loss)))


def PlayAGame(display_games=True):
    tester = AI_Player(display_games)
    state = tester.env.state
    count = 0
    start = time.perf_counter()
    step = 0
    first_loss = 0
    while True:
        count += 1
        step += 1
        action = tester.get_action(state)
        next_state, terminal, reward = tester.do_step(action)
        state = next_state
        print(reward)
        time.sleep(0.5)
        if terminal:
            if reward == 1:
                print("HOORAY!!! YOU WON THE GAME")
            else:
                print("OOPS!!! YOU LOST THE GAME")
            tester.env.reset()
            step = 0
            state = tester.env.state
            break


main()
