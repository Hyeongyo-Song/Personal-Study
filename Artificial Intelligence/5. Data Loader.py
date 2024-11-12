import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class MultiVariableLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self, x):
        return self.linear(x)


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73,85,75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x,y

Dataset = CustomDataset()

dataLoader = DataLoader(
    Dataset,
    batch_size = 2,
    shuffle = True
)

model = MultiVariableLinearRegression()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

Epochs = 20
for epoch in range(Epochs):
    for batch_index, samples in enumerate(dataLoader):
        x_train, y_train = samples

        prediction = model(x_train)
        loss = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epochs {}/{} Batch {}/{} prediction {} Loss {}'.format(epoch, Epochs, batch_index, len(dataLoader), prediction, loss))