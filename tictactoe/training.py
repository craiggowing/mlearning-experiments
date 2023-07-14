import torch
import torch.nn.functional as F
from ttt_game import get_best_moves
from ttt_model import TTTModel
import random

def get_data():
    all_data = list(get_best_moves())
    random.shuffle(all_data)
    data_count = len(all_data)

    Xall = [t[0] for t in all_data]
    Yall = [t[1] for t in all_data]

    Xall = F.one_hot(torch.tensor(Xall).view(-1, 9), num_classes=3).view(-1, 27).float()
    Yall = torch.tensor(Yall)
    Xall = Xall.to("cuda")
    Yall = Yall.to("cuda")

    Xtrain = Xall[:int(data_count * 0.9)]
    Ytrain = Yall[:int(data_count * 0.9)]

    Xvalidate = Xall[int(data_count * 0.9):]
    Yvalidate = Yall[int(data_count * 0.9):]

    return Xall, Yall, Xtrain, Ytrain, Xvalidate, Yvalidate


def get_model():
    model = TTTModel().to("cuda")
    model.train()
    return model

def do_training():
    model = get_model()
    Xall, Yall, Xtrain, Ytrain, Xvalidate, Yvalidate = get_data()

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    with torch.no_grad():
        logits = model.forward(Xtrain)
        train_loss = loss_function(logits, Ytrain)
        logits = model.forward(Xvalidate)
        val_loss = loss_function(logits, Yvalidate)

    print(f"After 0 steps: Lt:{train_loss} Lv:{val_loss}")

    torch.save(model.state_dict(), 'model_weights.pth')
    for passes in range(1000):
        idxs = torch.tensor(random.sample(range(len(Xtrain)), 50))
        Xbatch = Xtrain[idxs]
        Ybatch = Ytrain[idxs]
        for _ in range(1000):
            model.zero_grad()
            logits = model.forward(Xbatch)  # Do the forward calculations for the batch (note the batch normalization)
            loss = loss_function(logits, Ybatch)
            loss.backward()  # Calculate the gradients based on the loss function
            optimizer.step()  # Perform the NN adjustments of the weights and biases
        train_loss = loss

        with torch.no_grad():
            logits = model.forward(Xvalidate)
            loss = loss_function(logits, Yvalidate)
        val_loss = loss

        print(f"After {(passes + 1) * 1000} steps: Lt:{train_loss} Lv:{val_loss}")
        torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    do_training()
