# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py


import random
import torch


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, p_previous: float = 0.5):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
            p_previous (float) -- with this probability a 'history' image will be returned
        """
        self.pool_size = pool_size
        self.p_previous = p_previous
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p < self.p_previous:  # with 'p_previous' chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by (1 - p_previous) chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


if __name__ == '__main__':
    create_tensor = lambda: torch.randn((1, 3, 64, 64))
    pool = ImagePool(pool_size=50)
    for _ in range(5):
        pool.query(create_tensor())
    x = create_tensor()
    y = pool.query(x)
    z = (x == y).to(dtype=torch.float).mean()
    t = 1