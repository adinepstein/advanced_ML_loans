import torch
import pandas as ps
import numpy as np
from src.parse_data import get_parsed_data

torch.manual_seed(1234)

BATCH_SIZE = 16
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


def jensen_shannon_divergence(y_true,y_pred):
    avg_y = (y_true+ y_pred)/2.0
    jsd = 0.5 * kl_divergence(y_true,avg_y) + 0.5* kl_divergence(y_pred,avg_y)
    return jsd


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def calculate_top_k_prob(array,k=5):
    vec_sort = np.sort(array)[-1::-1]
    topk = vec_sort[:k]
    ary = np.arange(k)
    return np.prod([np.exp(topk[i]) / np.sum(np.exp(topk[i:])) for i in ary])


def listnet_loss(true_array,pred_array,k=5):
    loss = - np.sum(calculate_top_k_prob(true_array,k)* np.log(calculate_top_k_prob(pred_array, k)))
    return loss


def train_model(train_x_data,train_y_data,val_x_data,val_y_data):
    model=ListNet(INPUT_DIM,HIDDEN_LAYER_1,HIDDEN_LAYER_2)
    criterion = listnet_loss
    optimizer = torch.optim.SGD(model.parameters())
    train_loss_list=[]
    validation_loss_list = []
    train_acc =[]
    validation_acc = []
    for epoch in range(EPOCHS):
        print(f"start epoch {epoch+1}")
        train_size=len(train_x_data)
        val_size = len(val_x_data)
        suffle_train=np.random.permutation(train_size)
        suffle_val=np.random.permutation(val_size)
        train_loss = 0
        for i in range(0,train_size,BATCH_SIZE):
            try:
                x=torch.tensor(np.asarray(train_x_data[suffle_train[i:i+BATCH_SIZE]]))
                y_true=torch.tensor(np.asarray(train_y_data[suffle_train[i:i+BATCH_SIZE]]))
                y_pred = model(x)
                loss = criterion(y_true, y_pred)
                train_loss += loss*len(y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                pass
        val_loss=0
        for i in range(0,val_size,BATCH_SIZE):
            try:
                x = torch.tensor(np.asarray(train_x_data[suffle_val[i:i + BATCH_SIZE]]))
                y_true = torch.tensor(np.asarray(train_y_data[suffle_val[i:i + BATCH_SIZE]]))
                y_pred = model(x)
                loss = criterion(y_true,y_pred)
                val_loss+=loss*len(x)

            except:
                pass
        train_loss_list.append(train_loss/train_size)
        validation_loss_list.append(val_loss/val_size)
        print(f"epoch {epoch+1} : train loss - {train_loss/train_size}, val loss - {val_loss/val_size}")



def predict():
    pass


if __name__ == '__main__':
    data = get_data()
    print(data.shape)
    train_data, val_data, test_data = split_data_to_train_validation_test(data, 0.1, 0.05)
    print(f"train_data size - {train_data.shape}, val size - {val_data.shape}, test size - {test_data.shape}")
