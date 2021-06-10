import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, Dropout, Activation, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Cropping2D, Concatenate
from tensorflow.keras.models import Model

class DownConvBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ksize, pad_type='valid', drop_rate=0.2, activation='relu', **kwarg):
        super(DownConvBlock, self).__init__()

        self.conv2d_1 = Conv2D(channels, (ksize, ksize), padding=pad_type)
        self.drop_1 = Dropout(drop_rate)
        self.activation_1 = Activation(activation)
        
        self.conv2d_2 = Conv2D(channels, (ksize, ksize), padding=pad_type)
        self.drop_2 = Dropout(drop_rate)
        self.activation_2 = Activation(activation)

    def call(self, x, training=True):
        ## first
        x = self.conv2d_1(x)
        if training:
            x = self.drop_1(x)
        x = self.activation_1(x)

        ## second
        x = self.conv2d_2(x)
        if training:
            x = self.drop_2(x)
        x = self.activation_2(x)

        return x

class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ksize, strides=2, pad_type='valid', drop_rate=0.2, activation='relu', **kwarg):
        super(UpConvBlock, self).__init__()

        self.upconv = Conv2DTranspose(channels, ksize, strides=strides, padding=pad_type)
        self.drop = Dropout(drop_rate)
        self.activation = Activation(activation)

    def call(self, x, training=True):
        x = self.upconv(x)
        if training:
            x = self.drop(x)
        x = self.activation(x)

        return x

def CropNConcat(x, y):
    xshape = x.shape
    yshape = y.shape
    row_diff = (yshape[1] - xshape[1]) // 2
    col_diff = (yshape[2] - xshape[2]) // 2
    ## crop
    if row_diff%2 == 0:
        cropped = Cropping2D(cropping=(row_diff, col_diff))(y)
    else:
        cropped = Cropping2D(cropping=((row_diff, row_diff+1), (col_diff, col_diff+1)))(y)
    ## concat
    out = Concatenate()([x, cropped])

    return out

def UNet(input_shape, root_filter=64, layer_depth=5, drop_rate=0.2, training=True):
    ## setting
    filter_list = [root_filter*(2**i) for i in range(layer_depth)]

    input_layer = Input(shape=input_shape)
    x = input_layer

    skip_connects = []
    for i in range(layer_depth-1):
        x = DownConvBlock(filter_list[i], 3)(x)
        skip_connects.append(x)
        x = MaxPooling2D((2, 2))(x)
    ## Last
    x = DownConvBlock(filter_list[-1], 3)(x)

    for i in reversed(range(layer_depth-1)):
        x = UpConvBlock(filter_list[i], 2, strides=2)(x)
        x = CropNConcat(x, skip_connects[i])
        x = DownConvBlock(filter_list[i], 3)(x)

    out = Conv2D(1, (1, 1), strides=1)(x)

    model = Model(input_layer, out)

    return model

if __name__ == '__main__':
    model = UNet((120, 120, 3), layer_depth=3)

    model.summary()
    tf.keras.utils.plot_model(model)

