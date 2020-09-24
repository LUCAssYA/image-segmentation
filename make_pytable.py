import tables
import os, sys
from glob import glob
import PIL
import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches
import random
import copy
from keras.utils import to_categorical
from PIL import Image
import stain_normalization


def make_pytable(img_path, label_path, patch_size, stride_size, pad_size, split, num_classes, imgtype, labeltype):
    img_dtype = {}
    img_dtype['mask'] = tables.UInt8Atom()
    img_dtype['img'] = tables.Float32Atom()

    train_file = glob(img_path+"*."+imgtype)
    num_train = int(len(train_file)*(1-split))
    val_file = train_file[num_train:]
    train_file = train_file[:num_train]

    phases = {}
    phases['train'], phases['val'] = train_file, val_file

    block_shape = {}
    block_shape['img'] = np.array((patch_size, patch_size, 3))
    block_shape['mask'] = np.array((patch_size, patch_size, num_classes))

    filters = tables.Filters(complevel = 6, complib = 'zlib')

    storage = {}

    imgtypes = ['img', 'mask']

    for phase in phases.key():
        print(phase)

        table_file = tables.open_file(f"./table_{phase}.pytable", mode = 'w')

        for type in imgtypes:
            storage[type] = table_file.create_earray(table_file.root, type, img_dtype[type],
                                                     shape = np.append([0], block_shape[type]),
                                                     chunkshape= np.append([1], block_shape[type]),
                                                     filters = filters)

        for f in phases[phase]:
            print(f)

            for type in imgtypes:
                if type == "img":
                    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                    img = stain_normalization.normalizeStaining(img)
                    img = img/255.0

                    img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)])
                    img = extract_patches(img, (patch_size, patch_size, 3), stride_size)

                    img = img.reshape(-1, patch_size, patch_size, 3)

                else:
                    img = cv2.cvtColor(cv2.imread(label_path+f.replace(imgtype, labeltype)), cv2.IMREAD_GRAYSCALE)

                    if num_classes > 1:
                        img = to_categorical(img, num_classes = num_classes)
                    else:
                        img = img.reshape(img.shape[0], img.shape[1], 1)
                img = padAndPatch(img, pad_size, patch_size, stride_size)

                storage[type].append(img)
        table_file.close()

def padAndPatch(img, pad_size, patch_size, strides):
    img = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode = 'reflect')
    img = extract_patches(img, (patch_size, patch_size, img.shape[-1]), strides)
    img = img.reshape(-1, patch_size, patch_size, img.shape[-1])

    return img









