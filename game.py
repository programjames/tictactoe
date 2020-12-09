import numpy as np

WIN_POSITIONS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
class Game(object):
    @staticmethod
    def can_move(pos, move):
        p = list(pos)
        p[move] = 1
        if sum(p) == sum(pos)+0.5:
            return p
        return False
        
    def __init__(self, order=0):
        self.board = np.full(9, 0.5)
        self.order = order # who goes first; 0 means player1, 1 means player2.
    def update(self, player1, player2):
        m = 1
        if self.order:
            m = -1
            player1, player2 = player2, player1
        move = player1.move(self.board)
        if self.board[move] != 0.5:
            return -1*m # player2 wins.
        else:
            self.board[move] = 1
            for pos in WIN_POSITIONS:
                for i in pos:
                    if self.board[i] != 1:
                        break
                else:
                    return 1*m # player1 wins.
        
        if all(self.board != 0.5):
            return 0*m # it's a draw.
        
        move = player2.move(1-self.board) # invert the x/o numbers.
        if self.board[move] != 0.5:
            return 1*m # player1 wins.
        else:
            self.board[move] = 0
            for pos in WIN_POSITIONS:
                for i in pos:
                    if self.board[i] != 0:
                        break
                else:
                    return -1*m # player2 wins.
        return None # no one wins.
    def copy_board(self):
        return self.board[:]
    def print(self):
        printable = np.resize(self.board, (3,3))
        for r in printable:
            s = "".join("O" if i==0 else "X" if i==1 else " " for i in r)
            print(s)
        
class RandomPlayer(object):
    def __init__(self):
        pass
    def move(self, board):
        open = [i for i in range(9) if board[i]==0.5]
        move = np.random.choice(open)
        return move
        
def play_match(player1, player2):
    game = Game()
    while game.update(player1, player2) is None:
        game.print()
        print("---")
    game.print()
    print("---")
        
if __name__ == "__main__":
    play_match(RandomPlayer(), RandomPlayer())
