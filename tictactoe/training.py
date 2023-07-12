import torch
import torch.nn.functional as F

TRAINING_DATA = (
    (((0, 0, 0),
      (0, 0, 0),
      (0, 0, 0)), 4),
    (((0, 0, 0),
      (0, 2, 0),
      (0, 0, 0)), 0),
)


Xall = [t[0] for t in TRAINING_DATA]
Yall = [t[1] for t in TRAINING_DATA]

Xall = F.one_hot(torch.tensor(Xall).view(-1, 9), num_classes=3).view(-1, 27).float()
Yall = torch.tensor(Yall)

model = TTTModel().to("cuda")
model.train()
Xall = Xall.to("cuda")
Yall = Yall.to("cuda")

lr = 0.1
for _ in range(10000):
    model.zero_grad()
    logits = model.forward(Xall)
    loss = F.cross_entropy(logits, Yall)
    loss.backward()
    for p in model.parameters():
        p.data += -lr * p.grad
