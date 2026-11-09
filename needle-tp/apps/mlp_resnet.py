import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      norm(hidden_dim),
      nn.ReLU(),
      nn.Dropout(drop_prob),
      nn.Linear(hidden_dim, dim),
      norm(dim),
    )

    return nn.Sequential(nn.Residual(fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [
      nn.Linear(dim, hidden_dim),
      nn.ReLU(),
    ]
    for _ in range(num_blocks):
      modules.append(
        ResidualBlock(
          dim=hidden_dim,
          hidden_dim=hidden_dim // 2,
          norm=norm,
          drop_prob=drop_prob,
        )
      )
    modules.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = nn.SoftmaxLoss()

    # Set model mode
    if opt is not None:
      model.train()
    else:
      model.eval()

    total_loss = 0.0
    total_error = 0.0
    total_samples = 0

    for X, y in dataloader:
      batch_size = X.shape[0]
      X_flat = X.reshape((batch_size, -1))

      if opt is not None:
        opt.reset_grad()
        logits = model(X_flat)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
      else:
        logits = model(X_flat)
        loss = loss_fn(logits, y)

      total_loss += loss.numpy() * batch_size

      preds = logits.numpy().argmax(axis=1)
      total_error += np.sum(preds != y.numpy())
      total_samples += batch_size

    avg_error = total_error / total_samples
    avg_loss = total_loss / total_samples

    return avg_error, float(avg_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )

    train_loader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
      train_err, train_loss = epoch(train_loader, model, opt=opt)
      test_err, test_loss = epoch(test_loader, model, opt=None)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
