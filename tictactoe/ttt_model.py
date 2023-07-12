import torch


class TTTModel(torch.nn.Module):
    """
    Inputs:
    (9, 3) - 9 x 3 one hot vectors [EMPTY, MY_PIECE, OPPONENT_PIECE]

    Outputs:
    (9) - logits for where to place piece
    """


    def __init__(self):
        super(TTTModel, self).__init__()

        # 27 inputs, 50 hidden
        self.linear1 = torch.nn.Linear(27, 50)
        self.activation = torch.nn.Tanh()
        # 50 hidden to 9 output
        self.linear2 = torch.nn.Linear(50, 9)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
