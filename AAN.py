import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Add, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Lambda, PReLU, Activation, Multiply
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer

class AttentionBranch(Layer):
    def __init__(self, chs, ksize=3):
        super(AttentionBranch, self).__init__()

        ## 
        self.k1 = Conv2D(chs, (3, 3), padding='same', use_bias=False)
        self.lrelu = LeakyReLU(alpha=0.2)
        self.k2 = Conv2D(chs, (1, 1), padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')
        ##
        self.k3 = Conv2D(chs, (3, 3), padding='same', use_bias=False)
        ##
        self.k4 = Conv2D(chs, (3, 3), padding='same', use_bias=False)

    def call(self, x):

        yy = self.k1(x)
        yy = self.lrelu(yy)
        yy = self.k2(yy)
        yy = self.sigmoid(yy)
        ##
        yy2 = self.k3(x)
        ##
        out = Multiply()([yy, yy2])
        out = self.k4(out)

        return out

class ADM(Layer):
    def __init__(self, chs, K=2):
        super(ADM, self).__init__()

        reduction = 4
        self.g_avg_pool = GlobalAveragePooling2D()
        self.d1 = Dense(chs // reduction, use_bias=False)
        self.relu = Activation('relu')
        self.d2 = Dense(K, activation='softmax', use_bias=False)
        
    def call(self, x):
        
        yy = self.g_avg_pool(x)
        yy = self.d1(yy)
        yy = self.relu(yy)
        yy = self.d2(yy)

        return yy


class AAB(Layer):
    def __init__(self, chs):
        super(AAB, self).__init__()

        ## Attention Dropout Module
        self.ADM = ADM(chs, K=2)
        # first 1x1 conv
        self.first_conv = Conv2D(chs, (1, 1), use_bias=False, padding='same')
        self.lrelu = LeakyReLU(0.2)
        ## attention branch
        self.AttnB = AttentionBranch(chs)
        ## non-attention branch ##TODO: activation?
        self.non_attn = Conv2D(chs, (3, 3), use_bias=False, padding='same', activation=None)

        ## out conv 1x1
        self.last_conv = Conv2D(chs, (1, 1), use_bias=False, padding='same')

    def call(self, x):
        residual = x

        ADM_out = self.ADM(x)
        ##
        yy = self.first_conv(x)
        yy = self.lrelu(yy)

        yy_attn = self.AttnB(yy)
        yy_non_attn = self.non_attn(yy)
        ##
        ADM_1 = Lambda(lambda x: tf.slice(x, [0, 0], [-1, 1]))(ADM_out)
        ADM_2 = Lambda(lambda x: tf.slice(x, [0, 1], [-1, 1]))(ADM_out)
        ## or directly use tf.slice on ADM_out
        # ADM_1 = tf.slice(ADM_out, [0, 0], [-1, 1])
        # ADM_2 = tf.slice(ADM_out, [0, 1], [-1, 1])

        yy_attn = Multiply()([yy_attn, ADM_1])
        yy_non_attn = Multiply()([yy_non_attn, ADM_2])

        out = Add()([yy_attn, yy_non_attn])
        out = self.lrelu(out)

        out = self.last_conv(out)
        out = Add()([residual, out])

        return out

class PA(Layer):
    def __init__(self, upchs):
        super(PA, self).__init__()
        self.conv = Conv2D(upchs, (1, 1), activation='sigmoid')

    def call(self, x):
        y = self.conv(x)
        out = Multiply()([y, x])

        return out

class Reconstruct(Layer):
    def __init__(self, chs, input_shape, upchs=16):
        super(Reconstruct, self).__init__()
        self.upscale = Lambda(lambda x: tf.image.resize(x, [input_shape[0]*2, input_shape[1]*2], method='nearest'))
        self.upscal2 = Lambda(lambda x: tf.image.resize(x, [input_shape[0]*4, input_shape[1]*4], method='nearest'))
        self.lrelu = LeakyReLU(0.2)
        ### x2
        self.k1 = Conv2D(upchs, (3, 3), padding='same')
        self.att1 = PA(upchs)
        self.HRconv1 = Conv2D(upchs, (3, 3), padding='same')
        ### x2
        self.k2 = Conv2D(upchs, (3, 3), padding='same')
        self.att2 = PA(upchs)
        self.HRconv2 = Conv2D(upchs, (3, 3), padding='same')


    def call(self, x):
        ### x2
        y = self.upscale(x)
        y = self.k1(y)
        y = self.lrelu(self.att1(y))
        y = self.lrelu(self.HRconv1(y))
        #### x4
        y = self.upscal2(y)
        y = self.k2(y)
        y = self.lrelu(self.att2(y))
        y = self.lrelu(self.HRconv2(y))

        return y

        
def A2N(input_shape, out_channel, chs=32):
    x = Input(shape=input_shape)

    ## Frist conv2D 3x3, feature extraction
    x2 = Conv2D(chs, (3, 3), padding='same', name='FeatureExtract')(x)
    fea = x2
    ## AAB block
    for i in range(3):
        x2 = AAB(chs)(x2)
    ## Add
    AAB_out = Conv2D(chs, (3, 3), padding='same', name='A2B_Conv')(x2)
    fea = Add(name='Add_Fea_A2B')([fea, AAB_out])

    ## reconstruction part x4 scale
    recon_out = Reconstruct(chs, input_shape, upchs=16)(fea)
    
    ## conv last
    out = Conv2D(out_channel, (3, 3), padding='same', name='LastConv')(recon_out)

    ## LR bilinear interpolation
    ILR = Lambda(lambda x: tf.image.resize(x, [input_shape[0]*4, input_shape[1]*4]), name='LRInterpo')

    ## output layer add
    out = Add(name='FinalOutput')([out, ILR(x)])

    model = Model(x, out)

    return model

if __name__ == '__main__':
    import numpy as np
    model = A2N((120, 120, 1), 1)
    model.summary()

    test_input = np.random.random(size=(1, 120, 120, 1))
    out = model(test_input)

    print(out.shape)
