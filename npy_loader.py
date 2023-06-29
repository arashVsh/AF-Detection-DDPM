import torch
from torch.utils.data import TensorDataset, DataLoader
from Settings import BATCH_SIZE


def custom_loader(samples):
    tensor = torch.Tensor(samples)  # transform to torch tensor
    my_dataset = TensorDataset(tensor)  # create your datset
    my_dataloader = DataLoader(
        my_dataset, batch_size=BATCH_SIZE, shuffle=True
    )  # create your dataloader
    return my_dataloader
