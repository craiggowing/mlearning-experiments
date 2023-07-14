from copy import deepcopy

class TTTGame:
    def __init__(self, board=None, turn = 1):
        if board == None:
            self.board = [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]
        else:
            self.board = deepcopy(board)
        self.turn = turn

    def move(self, row, col):
        # It is always player 1's turn.
        # After player 1 moves we reverse pieces and player 1 goes
        # with the other set of pieces now!
        new_game = TTTGame(self.board, turn = self.turn)
        new_game.board[row][col] = new_game.turn
        new_game.turn = 1 if new_game.turn == 2 else 2
        return new_game

    def get_winner(self):
        win_options = [
            [cell for cell in self.board[0]],
            [cell for cell in self.board[1]],
            [cell for cell in self.board[2]],
            [self.board[row][0] for row in range(3)],
            [self.board[row][1] for row in range(3)],
            [self.board[row][2] for row in range(3)],
            [self.board[i][i] for i in range(3)],
            [self.board[i][2 - i] for i in range(3)],
        ]
        for option in win_options:
            if all(o == 1 for o in option):
                return 1
            if all(o == 2 for o in option):
                return 2
        if not self.valid_spaces():
            return 0  # Draw
        return None  # No winner

    def valid_spaces(self):
        res = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 0:
                    res.append((row, col))
        return res

def get_best_moves():
    """
    We want to get a series of moves which results in either a draw, or a win
    for player 1 (the current player) since the NN will always view the game
    from the perspective of player 1. For Tic-Tac-Toe we can easily just play
    every possible game and reverse any of the losing games.
    """
    game_data = set()

    def next_move(game):
        winner = game.get_winner()
        if winner is not None:
            return {winner}
        next_wins = set()
        future_wins = set()
        future_draws_and_wins = set()
        future_draws = set()
        results = set()
        for row, col in game.valid_spaces():
            next_game = game.move(row, col)
            next_results = next_move(next_game)
            results |= next_results
            if game.turn == 2 and next_game.get_winner() == 2:
                # If Player 2 can win with their move, reject this line
                return {2}
            elif game.turn == 1 and next_game.get_winner() == 1:
                # If Player 1 can win with their move, only accept these lines
                next_wins.add((tuple([tuple(row) for row in game.board]), row * 3 + col))
            elif game.turn == 1:
                if next_results == {1}:
                    # Player 1 will only win in this future
                    future_wins.add((tuple([tuple(row) for row in game.board]), row * 3 + col)) 
                elif next_results == {0, 1}:
                    # Player 1 could win or draw in this future
                    future_draws_and_wins.add((tuple([tuple(row) for row in game.board]), row * 3 + col))
                elif next_results == {0}:
                    # Player 1 could only draw in this future
                    future_draws.add((tuple([tuple(row) for row in game.board]), row * 3 + col))
        # We only accept the draws if we don't have any just wins
        if game.turn == 2:
            return results

        if next_wins:
            game_data.update(next_wins)
            return {1}
        elif future_wins:
            game_data.update(future_wins)
            return {1}
        elif future_draws_and_wins:
            game_data.update(future_draws_and_wins)
            return {0, 1}
        elif future_draws:
            game_data.update(future_draws)
            return {0}
        return set()

    next_move(TTTGame())
    next_move(TTTGame(turn=2))
    return tuple(game_data)

def print_moves(moves = None):
    if moves is None:
        moves = get_best_moves()
    for board, position in moves:
        buffer = ""
        for row in range(3):
            for col in range(3):
                if row * 3 + col == position:
                    buffer += "X"
                    if board[row][col] != 0:
                        buffer += "!"
                elif board[row][col] == 0:
                    buffer += "_"
                elif board[row][col] == 1:
                    buffer += "x"
                elif board[row][col] == 2:
                    buffer += "o"
                else:
                    buffer += "?"
            buffer += "\n"
        print(buffer)

