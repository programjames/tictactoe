import numpy as np
from tensorflow import keras
from game import Game

class NNPlayer(object):
    def __init__(self):
        self.q_model = keras.models.load_model('q_model.k')
    def move(self, board):
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

def prettyprint(board):
    board = np.resize(board, (3, 3))
    i = 0
    s = []
    for row in board:
        s.append(" | ".join("O" if j==0 else "X" if j==1 else str(i+k) for k,j in enumerate(row)))
        i += 3
    print("\n---------\n".join(s))

class HumanPlayer(object):
    def __init__(self):
        pass
    def move(self, board):
        prettyprint(board)
        return int(input("Move?"))
        

def play_match(player1, player2):
    game = Game()
    while game.update(player1, player2) is None:
        print("\n\n")
        prettyprint(game.board)
        print("\n\n")
    prettyprint(game.board)

play_match(HumanPlayer(), NNPlayer())#, HumanPlayer())
