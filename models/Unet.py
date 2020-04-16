from keras.layers import Conv2D, MaxPool2D, Dense, Input, UpSampling2D, concatenate, Reshape
from keras.models import Model

def down_block(x, filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'):
    c = Conv2D(filters, kernel_size, activation='relu', padding = padding)(x)
    c = Conv2D(filters, kernel_size, activation = 'relu', padding = padding)(c)
    p = MaxPool2D(pool_size=(2, 2))(c)
    return c, p

def up_block(x, s, filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'):
    up = UpSampling2D((2, 2))(x)
    concat = concatenate([up, s], axis = 3)
    c = Conv2D(filters, kernel_size, activation='relu', padding = padding)(concat)
    c = Conv2D(filters, kernel_size, activation='relu', padding = padding)(c)

    return c

def bottlenect(x, filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_normal'):

    c = Conv2D(filters, kernel_size, activation='relu', padding = padding)(x)
    c = Conv2D(filters, kernel_size, activation='relu', padding=padding)(c)
    return c

def Unet(input_size = (512, 512, 3), classes = 9, wf = 6, depth = 4):
    inputs = Input(input_size)
    conv_stack = []
    conv, pool = down_block(inputs, 2**wf)
    conv_stack.append(conv)

    for i in range(1, depth):
        conv, pool = down_block(pool, 2**(wf+i))
        conv_stack.append(conv)

    conv5 = bottlenect(pool, 2**(wf+depth))

    up = up_block(conv5, conv_stack.pop(), 2**(wf+depth-1))

    for i in range(depth-2, -1, -1):
        up = up_block(up, conv_stack.pop(), 2**(wf+i))

    output = Conv2D(classes, 1, padding = 'same', activation = 'softmax')(up)
    model = Model(inputs, output)
    model.summary()

    return model
