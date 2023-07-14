import torch
import torch.nn.functional as F
from ttt_game import TTTGame
from ttt_model import TTTModel

model = TTTModel().to("cuda")
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

def get_input(prompt, expected):
    while True:
        key = input(prompt).strip()
        if key in expected:
            return key

who_goes_first = get_input("Go first or second [12]: ", ["1", "2"])
# Switch these around since player 1 is the NN, you are player 2!
turn = 2 if who_goes_first == "1" else 1

game = TTTGame(turn = turn)

while game.get_winner() is None:
    if game.turn == 1:
        # This is the NNs turn, we need to sample based on the current game
        x = F.one_hot(torch.tensor(game.board).view(-1, 9), num_classes=3).view(-1, 27).float().to("cuda")
        logits = model.forward(x)[0]
        # Sort the logits by weight and find the idxs. We try them in decending order of valid moves
        moves = sorted(enumerate(logits), key=lambda a: -a[1])
        for move_idx, _ in moves:
            row = move_idx // 3
            col = move_idx % 3
            if game.board[row][col] != 0:
                continue
            game = game.move(row, col)
            print(f"NN moved to row:{row}, col:{col}")
            break
    elif game.turn == 2:
        while True:
            move_idx = get_input(("Which space [012]\n"
                                  "            [345]\n"
                                  "            [678]:"), [c for c in "0123456789"])
            row = int(move_idx) // 3
            col = int(move_idx) % 3
            if game.board[row][col] == 0:
                break
        game = game.move(row, col)
        print(f"Player moved to row:{row}, col:{col}")

    buffer = ""
    for row in range(3):
        for col in range(3):
            if game.board[row][col] == 0:
                buffer += "_"
            elif game.board[row][col] == 1:
                buffer += "x"
            elif game.board[row][col] == 2:
                buffer += "o"
            else:
                buffer += "?"
        buffer += "\n"
    print(buffer)

winner = game.get_winner()
if winner == 0:
    print("It was a draw!")
elif winner == 1:
    print("The NN won!")
elif winner == 2:
    print("The player won!")
