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

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(27, 54, bias=False),  # No bias due to BatchNorm1d layer
            torch.nn.BatchNorm1d(54),
            torch.nn.ReLU(),
            torch.nn.Linear(54, 9),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)
