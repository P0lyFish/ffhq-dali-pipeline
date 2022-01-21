import torch_dataset
import torch
import dali_dataset
import numpy as np
import kornia
from kornia.augmentation import AugmentationBase2D, AugmentationSequential

import time

NUM_THREADS = 8
IMAGE_DIR = 'images1024x1024'
BATCH_SIZE = 64


class RandomGaussianBlur(AugmentationBase2D):
    def __init__(self, kernel_size_limit=7, sigma=None, p=0.5):
        super(RandomGaussianBlur, self).__init__(p=p)
        self.kernel_size_limit = kernel_size_limit
        self.sigma = sigma

    def generate_parameters(self, input_shape):
        kernel_size = np.random.randint(1, self.kernel_size_limit // 2) * 2 + 1
        if self.sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        return {'kernel_size': (kernel_size, kernel_size), 'sigma': (sigma, sigma)}

    def compute_transformation(self, inp, params):
        return self.identity_matrix(inp)

    def apply_transform(self, inp, params, transform=None):
        return kornia.filters.gaussian_blur2d(inp, params['kernel_size'], params['sigma'])


class RandomMotionBlur(AugmentationBase2D):
    def __init__(self, kernel_size_limit=7, p=0.5):
        super(RandomMotionBlur, self).__init__(p=p)
        self.kernel_size_limit = kernel_size_limit

    def generate_parameters(self, input_shape):
        kernel_size = np.random.randint(1, self.kernel_size_limit // 2) * 2 + 1
        angle = torch.rand(input_shape[0]) * 360
        direction = torch.rand(input_shape[0]) * 2 - 1

        return {'kernel_size': kernel_size, 'angle': angle, 'direction': direction}

    def compute_transformation(self, inp, params):
        return self.identity_matrix(inp)

    def apply_transform(self, inp, params, transform=None):
        return kornia.filters.motion_blur(
            inp,
            kernel_size=params['kernel_size'],
            angle=params['angle'],
            direction=params['direction'],
        )


class RandomMedianBlur(AugmentationBase2D):
    def __init__(self, kernel_size_limit=7, p=0.5):
        super(RandomMedianBlur, self).__init__(p=p)
        self.kernel_size_limit = kernel_size_limit

    def generate_parameters(self, input_shape):
        kernel_size = np.random.randint(1, self.kernel_size_limit // 2) * 2 + 1

        return {'kernel_size': (kernel_size, kernel_size)}

    def compute_transformation(self, inp, params):
        return self.identity_matrix(inp)

    def apply_transform(self, inp, params, transform=None):
        return kornia.filters.median_blur(inp, params['kernel_size'])


class RandomBoxBlur(AugmentationBase2D):
    def __init__(self, kernel_size_limit=7, p=0.5):
        super(RandomBoxBlur, self).__init__(p=p)
        self.kernel_size_limit = kernel_size_limit

    def generate_paramters(self, input_shape):
        kernel_size = np.random.randint(1, self.kernel_size_limit // 2) * 2 + 1

        return {'kernel_size': (kernel_size, kernel_size)}

    def compute_transformation(self, inp, params):
        return self.identity_matrix(inp)

    def apply_transform(self, inp, params, transform=None):
        return kornia.filters.box_blur(inp, params['kernel_size'])


torch_dataloader = torch_dataset.get_dataloader(IMAGE_DIR, BATCH_SIZE, NUM_THREADS)
dali_dataloader = dali_dataset.get_dataloader(IMAGE_DIR, BATCH_SIZE, NUM_THREADS)

blur_transform = AugmentationSequential(
    RandomGaussianBlur(p=0.2),
    RandomMotionBlur(p=0.2),
    RandomMedianBlur(p=0.2),
    random_apply=1,
)

noise_transform = kornia.augmentation.RandomGaussianNoise(p=0.2)

print('Evaluation DALI pipeline')
start_time = time.time()
for i, data in enumerate(dali_dataloader):
    x = blur_transform(data[0]['image'])
    x = noise_transform(x)
end_time = time.time()
print(f'DALI + Kornia: {end_time - start_time}')

print('Evaluation torch dataloader')
start_time = time.time()
for i, data in enumerate(torch_dataloader):
    data[0].cuda()
    data[1].cuda()
    data[2].cuda()
end_time = time.time()
print(f'Torch + albumentations: {end_time - start_time}')
