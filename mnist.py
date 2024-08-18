from agrad import Module, one_hot_func, accuracy, F, Tensor
from agrad.nn import SoftmaxCrossEnt, Linear
from agrad.nn.optimizer import momentum
from agrad.utils import load_state_dict, save_state_dict
from agrad.vis import display
import csv
import os
import numpy as np

# 1. Load and process the data
file = "mnist_train_small.csv"
out = []
with open(file) as f:
    reader = csv.reader(f)
    out = list(reader)
data = np.array(out, dtype=int)
x = Tensor(data[:16000, 1:] / 255)
y = Tensor(one_hot_func(data[:16000, 0], 10))
x_val = Tensor(data[16000:, 1:] / 255)
y_val = Tensor(data[16000:, 0])


# 2. Define the model
class mlp(Module):
    def __init__(self):
        self.w = Linear(784, 96, kaiming=False)
        self.w1 = Linear(96, 10, kaiming=False)

    def forward(self, X):
        a1 = F.relu(self.w(X))
        a2 = F.tanh(self.w1(a1))
        return a2


# 3. Initialize the model and defines hyperparame3ter, loss and optimizer
model = mlp()
epochs = 15
BATCH_SIZE = 50
optimizer = momentum(lr=0.01)
loss = SoftmaxCrossEnt()


if "sd.pkl" in os.listdir():
    model = load_state_dict("sd.pkl", model)
else:
    for i in range(epochs):
        lsd = 0
        accu = 0

        for j in range(x.shape[0] // BATCH_SIZE):
            b_x = x[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
            b_y = y[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]

            model.zero_grad()
            yhat = model(b_x)
            ls = loss(yhat, b_y)
            lsd += ls
            loss.backward()
            optimizer.step(model)

        yhat = model.forward(x)

        print(
            f"Epoch {i}, Loss {'%.2f'%(lsd / (x.shape[0] // BATCH_SIZE))}, Accu: {'%.4f'%accuracy(yhat.data, y.data)}"
        )
    save_state_dict("sd.pkl", model)

yhat_val = model.forward(x_val)
print(
    f"Evaluation Accuracy: {(y_val.data == np.argmax(yhat_val.data,axis=1)).sum()/4000}"
)

while True:
    ind = int(input("Enter and index from 0 to 3999: "))
    if ind > 3999 or ind < 0:
        print("Bound Broken")
    else:
        display(x_val[ind].data.reshape((28, 28)))
        print(f"Predicted: {np.argmax(yhat_val[ind].data)}")
        print(f"Actual: {y_val[ind].data}")
