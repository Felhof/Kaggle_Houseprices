from functools import reduce
from itertools import chain
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class HouseNet(nn.Module):

    def __init__(self, input_, neurons, activations, out=1):
        super(HouseNet, self).__init__()


        def activation(name):
            return {
                "Id":None,
                "Sigmoid": nn.Sigmoid(),
                "ReLU": nn.ReLU()
            }[name]

        self._net = nn.Sequential(
            *filter(None, chain(*list(
                zip(
                    [nn.Linear(inp, outp) for inp, outp in zip([input_,*neurons],[*neurons,out])],
                    [activation(a) for a in activations]
                    )
                )))
            )

    def forward(self, input_):
        return self._net.forward(input_)


def fit_model(model, X, y):
    epochs = 10
    batch_size = 80
    learning_rate = 0.1

    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(np.array(X)), torch.Tensor(np.array(y)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    print("fitting model")
    for epoch in range(epochs):
        print("epoch: {}".format(epoch))
        for (x, ground_truth) in train_loader:
            output = torch.squeeze(model(x))
            loss = loss_function(output, ground_truth)
            print(loss.item())

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    print("done!")


def evaluate(model, X, y):
    predictions = torch.squeeze(model(torch.Tensor(np.array(X))))
    mse = nn.MSELoss()(predictions, torch.Tensor(np.array(y))).item()
    print("MSE: {}".format(mse))


if __name__ == "__main__":

    train_df=pd.read_csv("train.csv")
    evaluation_df=pd.read_csv("test.csv")

    #data_raw = pd.concat([train_df, evaluation_df], ignore_index=True, sort=False)
    #data_processed = preprocess(data_raw)
    #train_df, evaluation_df = np.split(data_processed, [len(train_df)])
    #evaluation_df.drop("SalePrice", axis="columns", inplace=True)

    train_features = train_df[["YearBuilt","PoolArea","GrLivArea","GarageArea"]]
    train_target = train_df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(train_features, train_target)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    model = HouseNet(4, [10,5],["Sigmoid","ReLU","Id"])
    fit_model(model, X_train, y_train)
    evaluate(model, X_test, y_test)
