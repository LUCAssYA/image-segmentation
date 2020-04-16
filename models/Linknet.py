from keras.models import Model

class Linknet:
    def __init__(self, classes):
        self.encoder_filter = [64, 128, 256, 512]
        self.decoder_nfilter = [256, 128, 64, 64]
        self.decoder_mfilter = [512, 256, 128, 64]
        self.encoder_block = []
        self.classes = classes
    def conv_bn(self, x, filter, kernel = 3,  strides = (1, 1), padding = 'same'):
        x = Conv2D(filter, kernel, strides = strides, padding = padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    def full_conv(self, x, filter, kernel = 3, strides = (2, 2), padding = 'same'):
        x = UpSampling2D((2, 2))(x)
        x = self.conv_bn(x, filter, kernel)

        return x

    def base(self, shape):
        input = Input(shape = shape)
        x = self.conv_bn(input, 64, 7, (2, 2), padding = 'same')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding = 'same')(x)

        for i in range(4):
            self.encoder_block.append(self.encoder(x, self.encoder_filter[i]))
            x = self.encoder_block[i]

        for i in range(4):
            x = self.decoder(x , self.decoder_mfilter[i], self.decoder_nfilter[i])
            if i != 3 : x = add([x, self.encoder_block[-2-i]])

        x = self.full_conv(x, 32, 3)

        x = self.conv_bn(x, 32, 3)

        x = self.full_conv(x, self.classes, 2)
        output = Activation('softmax')(x)

        model = Model(input, output)
        model.summary()

        return model

    def encoder(self, x, filter):
        en_conv1 = self.conv_bn(x, filter, 3, strides = (2, 2))
        en_conv2 = self.conv_bn(en_conv1, filter, 3)

        res = self.conv_bn(x, filter, 1, strides = (2, 2))

        add_conv1 = add([res, en_conv2])

        en_conv3 = self.conv_bn(add_conv1, filter, 3)
        en_conv4 = self.conv_bn(en_conv3, filter, 3)

        add_conv2 = add([add_conv1, en_conv4])

        return add_conv2

    def decoder(self, x, m, n):
        de_conv1 = self.conv_bn(x,int(m/4), 1)

        f_conv = self.full_conv(de_conv1, int(m/4), 3)

        de_conv2 = self.conv_bn(f_conv, n, 1)

        return de_conv2
