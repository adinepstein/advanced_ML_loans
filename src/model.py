import torch
import pandas as ps
import numpy as np
from src.parse_data import get_parsed_data

torch.manual_seed(1234)

BATCH_SIZE = 64
EPOCHS = 100
INPUT_DIM = 90
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 32


class ListNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim_1,hidden_dim_2):
        super(ListNet, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim_1)
        self.linear3 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear3 = torch.nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred = self.linear3(h2_relu).clamp(min=0)
        return y_pred


def get_data():
    df = get_parsed_data()
    print(f"Collection data - number of examples and features - {df.shape} ")
    data_array = df.to_numpy()
    return data_array


def split_data_to_train_validation_test(data_array, val_percentage, test_percentage):
    data_size = data_array.shape[0]
    train_set = data_array[:int(data_size * (1 - val_percentage - test_percentage))]
    val_set = data_array[int(data_size * (1 - val_percentage - test_percentage)):int(data_size * (1 - test_percentage))]
    test_set = data_array[int(data_size * (1 - test_percentage)):]
    return train_set, val_set, test_set


def train_model():
    pass


def predict():
    pass


if __name__ == '__main__':
    data = get_data()
    print(data.shape)
    train_data, val_data, test_data = split_data_to_train_validation_test(data, 0.1, 0.05)
    print(f"train_data size - {train_data.shape}, val size - {val_data.shape}, test size - {test_data.shape}")
