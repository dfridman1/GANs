from torch.utils.data import Dataset, DataLoader


def sample_random_batch(dataset: Dataset, batch_size: int):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )
    for batch in dataloader:
        return batch
