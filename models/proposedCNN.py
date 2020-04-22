from keras.layers import Conv2D,  BatchNormalization, Input
from keras.models import Model

class proposedCnn:
    def base(self, shape, classes):
        input = Input(shape= shape)

        conv = input
        for i in range(9):
            conv = Conv2D(64, 3, padding = 'same', activation = 'relu')(conv)
            conv = BatchNormalization()(conv)


        for i in range(2):
            conv = Conv2D(classes, 1, padding = 'same', activation = 'relu')(conv)
            conv = BatchNormalization()(conv)
        output = Conv2D(classes, 1, padding = 'same', activation = 'softmax')(conv)

        model = Model(input, output)
        model.summary()

        return model