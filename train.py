from keras.utils import HDF5Matrix
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import Unet, Linknet, ResUnet, proposedCNN
import os

def train(lr, patch_size, n_classes, epochs, gpuid, model_name, batch_size):
    if model_name == "Unet":
        model = Unet.Unet(input_size = (patch_size, patch_size, 3), classes=n_classes)
    elif model_name == "Linknet":
        model_c = Linknet.Linknet(n_classes)
        model = model_c.base((patch_size, patch_size, 3))
    elif model_name == "ResUnet":
        model_c = ResUnet.ResUnet()
        model = model_c.base((patch_size, patch_size, 3), n_classes)
    else:
        model_c = proposedCNN.proposedCnn()
        model = model_c.base((patch_size,patch_size, 3), n_classes)


    x_train = HDF5Matrix("table_train.pytable", 'img')
    y_train = HDF5Matrix("table_train.pytable", 'mask')
    x_val = HDF5Matrix("table_val.pytable", 'img')
    y_val = HDF5Matrix("table_val.pytable", 'mask')


    os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

    path = "checkpoint.hdf5"
    optimizer = Adam(lr = lr)
    checkpoint = ModelCheckpoint(filepath=path, monitor = 'val_loss', verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor = 'val_loss', patience=10, verbose = 1)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy' if n_classes > 1 else 'binary_crossentropy', metrics = ['acc'])
    model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs, callbacks=[checkpoint, earlystopping], validation_data=(x_val, y_val), shuffle = "batch")


    model.load_weights(path)
    model.save(model_name+" segmentation.hdf5")