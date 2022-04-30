import torch

from torch.nn.functional import one_hot


def conditional_input_encoder_generator(labels: torch.Tensor, cardinality: int):
    return one_hot(labels, cardinality).to(dtype=torch.float32)


def conditional_input_encoder_discriminator(labels: torch.Tensor, cardinality: int, spatial_size: int):
    batch_size = labels.shape[0]
    x = one_hot(labels, cardinality).to(dtype=torch.float32).view(batch_size, cardinality, 1, 1)
    x = x.repeat(1, 1, spatial_size, spatial_size)
    return x


if __name__ == '__main__':
    import numpy as np
    labels = np.asarray([1, 2, 4, 3])
    labels = torch.from_numpy(labels).to(dtype=torch.long)
    cardinality = 5
    x = conditional_input_encoder_discriminator(labels, cardinality, spatial_size=10)
    print(x.shape)