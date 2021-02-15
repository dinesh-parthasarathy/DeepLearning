import torch as t
from sklearn.metrics import f1_score
import numpy as np
from model import ResNet


class Trainer:

    def __init__(self,
                 model: ResNet,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self.F1_score = []
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        self._model.zero_grad()  # reset the gradients
        output = self._model(x)  # propagate through the network
        loss = self._crit(output, y)  # calculate the loss
        loss.backward()  # compute gradient by backward propagation
        self._optim.step()  # update weights
        return loss  # return the loss

    def val_test_step(self, x, y):
        output = self._model(x)  # predict
        loss = self._crit(output, y)  # propagate through the network and calculate the loss and predictions
        return loss, output >= 0.5  # return the loss and the predictions

    def train_epoch(self):
        self._model.train()
        total_loss = 0
        batch_size = self._train_dl.batch_sampler.batch_size
        no_batches = len(self._train_dl)  # MARK: check whether it returns all features or batches
        for x, y in self._train_dl:  # iterate through the training set

            if self._cuda:  # if a gpu is given
                x = x.cuda()  # transfer the batch to "cuda()"
                y = y.cuda()

            total_loss += self.train_step(x, y)  # perform a training step

        return total_loss / (batch_size * no_batches)

    def val_test(self):
        self._model.eval()
        total_loss = 0
        batch_size = self._val_test_dl.batch_sampler.batch_size
        no_batches = len(self._val_test_dl)
        Y = t.tensor([]).cuda()
        OUT = t.tensor([]).cuda()
        with t.no_grad():  # disable gradient computation
            for x, y in self._val_test_dl:  # iterate through the validation set
                if self._cuda:
                    x = x.cuda()  # transfer the batch to "cuda()"
                    y = y.cuda()
                loss, output = self.val_test_step(x, y)
                total_loss += loss
                Y = t.cat((Y, y))
                OUT = t.cat((OUT, output))
            self.F1_score.append(f1_score(Y.cpu(), OUT.cpu(), average="macro"))
            print("Mean f1 score for EPOCH: ", self.F1_score[-1])
            return total_loss / (batch_size * no_batches)

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        ctr = 0
        F1_scores_highest = np.array([-1, -1, -1, -1, -1], dtype=float)  # 5 best F1 scores saved as checkpoints
        while True:
            for epoch in range(epochs):  # stop by epoch number
                train_loss.append(self.train_epoch())  # train for a epoch
                val_loss.append(self.val_test())  # calculate the loss on the validation set
                if len(val_loss) > 1:  # MARK: Is it better to use F1 score or validation loss for early stopping ?
                    if val_loss[-1] >= val_loss[-2]:  # validation loss does not decrease
                        ctr += 1
                    else:
                        ctr = 0
                if 0 < self._early_stopping_patience <= ctr:
                    break

                # Choose checkpoint file with least F1 score for overwriting
                pos = np.argmax(self.F1_score[-1] - F1_scores_highest)

                # save only if current F1 score is greater than any one of the previously saved checkpoints
                if self.F1_score[-1] - F1_scores_highest[pos] > 0:
                    F1_scores_highest[pos] = self.F1_score[-1]
                    self.save_checkpoint(pos)

            return train_loss, val_loss
