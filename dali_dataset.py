import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline

import numpy as np
import random
import os.path as osp


class ExternalInputIterator(object):
    def __init__(self, batch_size, image_dir):
        self.images_dir = image_dir
        self.batch_size = batch_size
        with open(osp.join(self.images_dir, "file_list.txt"), 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        random.shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        if self.i == self.n:
            raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            f = open(osp.join(self.images_dir, jpeg_filename), 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            self.i = self.i + 1

            if self.i == self.n:
                break

        return (batch,)


def get_dataloader(image_dir, batch_size, num_threads):
    eii = ExternalInputIterator(batch_size, image_dir)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    with pipe:
        images = fn.external_source(source=eii, num_outputs=1)
        decode = fn.decoders.image(images, device='mixed', output_type=types.RGB)
        p = fn.random.coin_flip(0.4)
        quality = fn.random.uniform(range=[30, 50], dtype=types.DALIDataType.INT32)
        decode_jpeg = p * fn.jpeg_compression_distortion(decode, device='gpu', quality=quality) + (1 - p) * decode
        decode_float = fn.cast(decode_jpeg, dtype=types.DALIDataType.FLOAT)
        pipe.set_outputs(decode_float)

    pipe.build()

    dali_iter = DALIGenericIterator(pipelines=[pipe], output_map=['image'],
                                    reader_name=None,
                                    auto_reset=True,
                                    dynamic_shape=False,
                                    last_batch_padded=True)

    return dali_iter
