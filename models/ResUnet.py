from keras.layers import MaxPool2D, Conv2D, Activation, Input, BatchNormalization, add, UpSampling2D, concatenate
from keras.models import Model

class ResUnet:
    def conv_bn(self, x, filter, kernel, strides = (1, 1), padding = 'same', activation = 'relu', dilation = (1, 1), Act = False):
        x = Conv2D(filter, kernel, strides = strides, padding = padding, dilation_rate = dilation)(x)
        if Act == True:
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

        return x

    def base(self, shape, classes):
        inputs = Input(shape = shape)
        conv1 = self.conv_bn(inputs, filter = 32, kernel = 1, dilation = (1, 1), strides = (1, 1))
        rb1 = self.resblock(conv1, 32, 3, [1, 3, 15, 31], strides = (1, 1))

        conv2 = self.conv_bn(rb1, filter = 64, kernel = 1, dilation = (1, 1), strides= (2, 2))
        rb2 = self.resblock(conv2, 64, 3, [1, 3, 15, 31], strides = (1, 1))

        conv3 = self.conv_bn(rb2, filter = 128, kernel = 1, dilation = (1, 1), strides = (2, 2))
        rb3 = self.resblock(conv3, 128, 3, [1, 3, 15], strides = (1, 1))

        conv4 = self.conv_bn(rb3, filter = 256, kernel = 1, dilation=(1, 1), strides = (2, 2))
        rb4 = self.resblock(conv4, 256, kernel = 3, dilation = [1, 3, 15], strides = (1, 1))

        conv5 = self.conv_bn(rb4, filter = 512, kernel=1, strides = (2, 2), dilation = (1, 1))
        rb5 = self.resblock(conv5, filter = 512, kernel = 3, dilation = [1], strides= (1, 1))

        conv6 = self.conv_bn(rb5, filter = 1024, kernel = 1, dilation = (1, 1), strides = (2, 2))
        rb6 = self.resblock(conv6, filter = 1024, kernel = 3, dilation = [1], strides = (1, 1))

        psp1 = self.PSPpool(rb6, 1024)

        psp1 = UpSampling2D((2, 2))(psp1)
        comb1 = self.combine(psp1, rb5, 1024)
        rb7 = self.resblock(comb1, filter = 512, kernel = 3, dilation = [1], strides = (1, 1))

        rb7 = UpSampling2D((2, 2))(rb7)
        comb2 = self.combine(rb7, rb4, 512)
        rb8 = self.resblock(comb2, filter = 256, kernel = 3, dilation = [1, 3, 15], strides = (1, 1))

        rb8 = UpSampling2D((2, 2))(rb8)
        comb3 = self.combine(rb8, rb3, 256)
        rb9 = self.resblock(comb3, filter = 128, kernel = 3, dilation = [1, 3, 15], strides = (1, 1))

        rb9 = UpSampling2D((2, 2))(rb9)
        comb4 = self.combine(rb9, rb2, 128)
        rb10 = self.resblock(comb4, filter = 64, kernel = 3, dilation = [1, 3, 15, 31], strides = (1, 1))

        rb10 = UpSampling2D((2, 2))(rb10)
        comb5 = self.combine(rb10, rb1, 64)
        rb11 = self.resblock(comb5, filter = 32, kernel = 3, dilation = [1, 3, 15, 31], strides = (1, 1))

        comb6 = self.combine(rb11, conv1, 32)

        psp2 = self.PSPpool(comb6, 32)

        output = Conv2D(classes, kernel_size = 1, activation = 'softmax')(psp2)

        model = Model(inputs, output)
        model.summary()

        return model



    def resblock(self, x, filter, kernel, dilation, strides):
        dilation_block = []
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        for i in dilation:
            d = self.conv_bn(x, filter = filter, kernel = kernel, strides = strides, dilation = (i, i), Act = True)
            dilation_block.append(Conv2D(filter, kernel, strides = strides, dilation_rate = (1, 1), padding = 'same')(d))

        if len(dilation) == 4:
            out = add([dilation_block[0], dilation_block[1], dilation_block[2], dilation_block[3]])
        elif len(dilation) == 3:
            out = add([dilation_block[0], dilation_block[1], dilation_block[2]])
        else:
            out = dilation_block[0]

        return out

    def PSPpool(self, x, filter):
        pool = []
        for i in range(4):
            k = 2**i
            p = MaxPool2D((k, k))(x)
            p = UpSampling2D((k, k))(p)
            pool.append(self.conv_bn(p, filter=int(filter/4), kernel = 1, strides = (1, 1), dilation = (1, 1)))
        return concatenate([pool[0], pool[1], pool[2], pool[3]], axis = 3)

    def combine(self, x, r, filter):
        x = Activation('relu')(x)
        comb = concatenate([x, r], axis = 3)
        comb = self.conv_bn(comb, filter = filter, kernel = 1, strides=(1, 1), dilation=(1, 1))

        return comb


