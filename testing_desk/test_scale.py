import torch

loss = torch.nn.BCELoss()
sig = torch.nn.Sigmoid()
y = []
for _ in range(1000):
    x = torch.Tensor([torch.distributions.Normal(0.5, 0.1).rsample()])
    x = sig(x)
    # print(x)
    tar = torch.zeros_like(x) # torch.randint(low=0, high=1, size=(1,))
    print(tar, type(tar))
    c_loss = loss(x, tar)
    y.append(c_loss)
    # print(c_loss)
print(torch.mean(torch.Tensor([y])))
