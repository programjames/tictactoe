# Yay! It works! ~75% winrate against a random player (ties count as 0.5/1).

from tensorflow import keras
from game import Game, RandomPlayer, play_match
import itertools
import random
import numpy as np

board_positions = []
for pos in itertools.product(*[(0,0.5,1)]*9):
    if sum(pos) == 4.5 or sum(pos) == 5:
        board_positions.append(pos)

class NNPlayer(object):
    def __init__(self):
        # q_table
        self.q_model = keras.models.Sequential()
        self.q_model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(9,)))
        self.q_model.add(keras.layers.Dense(64, activation='sigmoid'))
        self.q_model.add(keras.layers.Dense(64, activation='sigmoid'))
        self.q_model.add(keras.layers.Dense(16, activation='sigmoid'))
        self.q_model.add(keras.layers.Dense(1, activation=None))
        self.q_model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

        # model to determine the best move
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(9,)))
        self.model.add(keras.layers.Dense(64, activation='sigmoid'))
        self.model.add(keras.layers.Dense(64, activation='sigmoid'))
        self.model.add(keras.layers.Dense(16, activation='sigmoid'))
        self.model.add(keras.layers.Dense(9, activation=None))
        self.model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

        self.do_random = 1

    def move(self, board):
        self.last_board = board
        if self.do_random > random.random():
            open = [i for i in range(9) if board[i]==0.5]
            move = np.random.choice(open)
            return move
        boards = []
        inds = []
        for move in range(9):
            p = Game.can_move(board, move)
            if p:
                boards.append(p)
                inds.append(move)
        r = [0 for i in range(9)]
        if boards == []:
            return r
        q_values = self.q_model.predict([boards], steps=1)
        i = inds[q_values.argmax()]
        return i

    def best_move(self, pos):
        boards = []
        inds = []
        for move in range(9):
            p = Game.can_move(pos, move)
            if p:
                boards.append(p)
                inds.append(move)
        r = [0 for i in range(9)]
        if boards == []:
            return r
        q_values = self.q_model.predict([boards], steps=1)
        i = inds[q_values.argmax()]
        r[i] = 1
        return r

class Trainer(object):
    def __init__(self, discount=0.95, memory=1000, random_move=1, random_delta=0.01):
        self.nnplayer = NNPlayer()
        self.randomplayer = RandomPlayer()
        self.game = Game(int(random.random()*2))

        self.discount = discount
        self.x_data = []
        self.y_data = []
        self.index = 0
        self.memory = memory
        self.random_move = random_move
        self.random_delta = random_delta

    def get_data(self, otherplayer, datapoints=100):
        wins = 0
        total_games = 0
        while datapoints > 0:
            datapoints -= 1
            old_board = self.game.copy_board()
            res = self.game.update(self.nnplayer, otherplayer)
            reward = res
            if reward is None:
                reward = 0
            reward = reward
            new_q_value = self.nnplayer.q_model.predict([[self.game.board]])[0][0]
            target_q_value = (reward + self.discount * new_q_value)
            #print(new_q_value, old_board, reward, target_q_value)
            if len(self.x_data) < self.memory:
                self.x_data.append(old_board)
                self.y_data.append(target_q_value)
            else:
                self.x_data[self.index] = old_board
                self.y_data[self.index] = target_q_value
                self.index += 1
                self.index %= self.memory
            
            if res is not None:
                self.game = Game(int(random.random()*2))
            if res is not None:
                if res == 1:
                    wins += 1
                elif res == 0:
                    wins += 0.5
                total_games += 1
        return wins/total_games

    def update_q_model(self, batch_size=100):
        self.nnplayer.q_model.fit([self.x_data], [self.y_data], verbose=0, batch_size=batch_size)

    def update_model(self, batch_size=100):
        # less than 9000 of these, not too costly...
        indices = np.random.choice(len(board_positions), size=batch_size)
        y_data = [self.nnplayer.best_move(board_positions[i]) for i in indices]
        bp = [board_positions[i] for i in indices]
        self.nnplayer.model.fit([bp], [y_data], verbose=0)

    def update(self):
        win_ratio = self.get_data(RandomPlayer())
        self.update_q_model()
        self.update_model()
        self.random_move -= self.random_delta
        self.nnplayer.do_random = self.random_move
        return win_ratio

trainer = Trainer()
while True:
    print(trainer.update())
    print("Playing a match (random at {}):".format(trainer.random_move))
    play_match(RandomPlayer(), trainer.nnplayer)
