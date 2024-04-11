import time
import torch
import numpy as np
import sys
from DQNmodel import DQN, Buffer
from env import MineSweeper
from display import DisplayGame
from numpy import float32
from torch.autograd import Variable
from multiprocessing import Process
from torch import FloatTensor, LongTensor


class ModelTrain():

    def __init__(self, width, height, bomb_no, render_flag):
        self.width = width
        self.height = height
        self.bomb_no = bomb_no
        self.box_count = width * height
        self.env = MineSweeper(self.width, self.height, self.bomb_no)
        self.current_model = DQN(self.box_count, self.box_count)
        self.target_model = DQN(self.box_count, self.box_count)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=0.003, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.buffer = Buffer(100000)
        self.gamma = 0.99
        self.render_flag = render_flag
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.90
        self.reward_threshold = 0.12
        self.reward_step = 0.01
        self.batch_size = 4096
        self.tau = 5e-5
        self.log = open("./training_log_DQN.txt", 'w')
        if self.render_flag:
            self.DisplayGame = DisplayGame(self.env.state)

    def load_models(self, number):
        path = "./trained_checkpoints_DQN/trained_for_" + str(number) + ".pth"
        weights = torch.load(path)
        self.current_model.load_state_dict(weights['current_state_dict'])
        self.target_model.load_state_dict(weights['target_state_dict'])
        self.optimizer.load_state_dict(weights['optimizer_state_dict'])
        self.current_model.epsilon = weights['epsilon']

    def get_action(self, state, mask):
        state = state.flatten()
        mask = mask.flatten()
        action = self.current_model.act(state, mask)
        return action

    def do_step(self, action):
        i = int(action / self.width)
        j = action % self.width
        if self.render_flag:
            self.DisplayGame.state = self.env.state
            self.DisplayGame.draw()
            self.DisplayGame.bugfix()
        next_state, terminal, reward = self.env.choose(i, j)
        next_fog = 1 - self.env.fog
        return next_state, terminal, reward, next_fog

    def epsilon_update(self, avg_reward):
        if avg_reward > self.reward_threshold:
            self.current_model.epsilon = max(self.epsilon_min, self.current_model.epsilon * self.epsilon_decay)
            self.reward_threshold += self.reward_step

    def TD_Loss(self):
        state, action, mask, reward, next_state, next_mask, terminal = self.buffer.sample(self.batch_size)
        state = Variable(FloatTensor(float32(state)))
        mask = Variable(FloatTensor(float32(mask)))
        next_state = FloatTensor(float32(next_state))
        action = LongTensor(float32(action))
        next_mask = FloatTensor(float32(next_mask))
        reward = FloatTensor(reward)
        done = FloatTensor(terminal)
        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss_print = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        for target_param, local_param in zip(self.target_model.parameters(), self.current_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        return loss_print

    def save_checkpoints(self, batch_no):
        path = "./trained_checkpoints_DQN/trained_for_" + str(batch_no) + ".pth"
        torch.save({
            'epoch': batch_no,
            'current_state_dict': self.current_model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.current_model.epsilon
        }, path)

    def save_logs(self, batch_no, avg_reward, loss, wins):
        res = [str(batch_no), "\tAvg Reward: ", str(avg_reward), "\t Loss: ", str(loss), "\t Wins: ", str(wins), "\t Epsilon: ", str(self.current_model.epsilon)]
        log_line = " ".join(res)
        print(log_line)
        self.log.write(log_line + "\n")
        self.log.flush()


def main():
    mazeWidth = 6
    mazeHeight = 6
    mazeBomb = 6
    driver = ModelTrain(mazeWidth, mazeHeight, mazeBomb, False)
    state = driver.env.state
    epochs = 20000
    save_every = 1000
    count = 0
    running_reward = 0
    batch_no = 0
    wins = 0
    total = 0
    while batch_no < epochs:
        mask = 1 - driver.env.fog
        action = driver.get_action(state, mask)
        next_state, terminal, reward, _ = driver.do_step(action)
        driver.buffer.push(state.flatten(), action, mask.flatten(), reward, next_state.flatten(), (1 - driver.env.fog).flatten(), terminal)
        state = next_state
        count += 1
        running_reward += reward
        if terminal:
            if reward == 1:
                wins += 1
            driver.env.reset()
            state = driver.env.state
            mask = driver.env.fog
            total += 1
        if count == driver.batch_size:
            driver.current_model.train()
            loss = driver.TD_Loss()
            driver.current_model.eval()
            batch_no += 1
            avg_reward = running_reward / driver.batch_size
            wins = wins * 100 / total
            driver.save_logs(batch_no, avg_reward, loss, wins)
            driver.epsilon_update(avg_reward)
            running_reward = 0
            count = 0
            wins = 0
            total = 0
            if batch_no % save_every == 0:
                driver.save_checkpoints(batch_no)


main()
