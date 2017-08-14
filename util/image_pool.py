import random
import numpy as np
import torch
from torch.autograd import Variable
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = [] # variable

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:# tensor
            image = torch.unsqueeze(image, 0) # no batch tensor
            if self.num_imgs < self.pool_size:
                # when pool unfull use current imgs, and add one in pool
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # pool is full, use random 50 history imgs as whole output
                # , 0.5 prob of replace randomly chosen one in pool by new G(x)
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        # return size of img with batchsize=pool_size
        return return_images
