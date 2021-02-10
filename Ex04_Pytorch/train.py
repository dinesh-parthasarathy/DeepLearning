import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
import torch.optim as opt
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data.csv', sep=';')
df_train, df_test = train_test_split(df, train_size=0.85, test_size=0.15)  # MARK: optimise train_test_split parameters(test size, train size, shuffle etc)

# set up data loading for the training and validation set
val_dl = t.utils.data.DataLoader(ChallengeDataset(df_test, 'val'), batch_size=100)
train_dl = t.utils.data.DataLoader(ChallengeDataset(df_train, 'train'), batch_size=100)  # MARK: optimise batch size

net = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss = t.nn.BCELoss()
opt1 = opt.Adam(net.parameters(), lr=1e-3)  # MARK: optimise lr

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(net, loss, opt1, train_dl, val_dl, cuda=True)

# go, go, go... call fit on trainer
res = trainer.fit(60)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
