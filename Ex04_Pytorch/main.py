import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

input = torch.randn(20, 3, 300, 300)
from model import ResNet
resobj = ResNet()
output = resobj(input)
print(output)

'''
train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

total = 0

for x, y in train_set:
    # print(y[0])
    break


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)  # apply softmax across second dim, 1st is a batch dimension

        return x


net = Net()
print(net)

X = torch.rand((28, 28))
X = X.view(-1, 28 * 28)
output = net(X)
print(output)

opt1 = opt.Adam(net.parameters(), lr=1e-3)

EPOCHS = 3
for epoch in range(EPOCHS):
    for data in train_set:
        # data is a batch of feature sets and labels
        x, y = data
        net.zero_grad()
        output = net(x.view(-1, 28 * 28))
        loss = F.nll_loss(output, y)
        loss.backward()
        opt1.step()
    print(loss)


with torch.no_grad():
    for data in train_set:
        x, y = data
        output = net(x.view(-1,28*28))
        for idx, i in enumerate(output):
            if torch.argmax() '''
