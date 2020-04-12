from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class HouseNet(nn.Module):

    def __init__(self, input_, neurons, activations, out=1):
        super(HouseNet, self).__init__()

        neurons.append(output)

        def activation(name):
            return {
                "Id":None
                "Sigmoid": nn.Sigmoid()
                "ReLU": nn.ReLU()
            }[name]

        self._net = Sequential(
            filter(None, chain(*list(
                zip(
                    [nn.Linear(inp, outp) for inp, out in zip([input_,*neurons],[*neurons,out])],
                    [activation(a) for a in activations]
                    )
                )))
            )

    def forward(self, input_):
        return self._net.forward(input_)

if __name__ == "__main__":

    train_df=pd.read_csv("train.csv")
    evaluation_df=pd.read_csv("test.csv")

    data_raw = pd.concat([train_df, evaluation_df], ignore_index=True, sort=False)
    data_processed = preprocess(data_raw)
    train_df, evaluation_df = np.split(data_processed, [len(train_df)])
    evaluation_df.drop("SalePrice", axis="columns", inplace=True)

    train_features = train_df.drop("SalePrice", axis="columns")
    train_target = train_df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(train_features, train_target)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
