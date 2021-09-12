import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2DTranspose, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import LeakyReLU, Reshape, Activation, Flatten

class DCGAN():
    def __init__(self):
        self.Gz = self.Generator()
        self.Dz = self.Discriminator()
    
        self.g_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        
        ## loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.batch_size = 64

    def Generator(self):
        init_ = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        ## 
        x = Input(shape=(100))
        g1 = Dense(16*512)(x)
        g1 = BatchNormalization()(g1)
        g1 = LeakyReLU()(g1)
        g1 = Reshape((4, 4, 512))(g1)
        ##
        #g2 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_)(g1)
        g2 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(g1)
        g2 = BatchNormalization()(g2)
        g2 = LeakyReLU()(g2)
        ##
        g3 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(g2)
        g3 = BatchNormalization()(g3)
        g3 = LeakyReLU()(g3)
        ##
        g4 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(g3)
        g4 = BatchNormalization()(g4)
        g4 = LeakyReLU()(g4)
        ##
        g5 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(g4)
        g5 = Activation('tanh')(g5)
    
        model = Model(inputs=x, outputs=g5)
    
        return model

    def Discriminator(self):
        init_ = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        ##
        x = Input(shape=(64, 64, 3))
        d1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
        d1 = BatchNormalization()(d1)
        d1 = LeakyReLU()(d1)
        ##
        d2 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(d1)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU()(d2)
        ##
        d3 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(d2)
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU()(d3)
        ##
        d4 = Conv2D(256, (3, 3), padding='same')(d3)
        d4 = BatchNormalization()(d4)
        d4 = LeakyReLU()(d4)
        ##
        d5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(d4)
        d5 = BatchNormalization()(d5)
        d5 = LeakyReLU()(d5)
        ##
        d6 = Flatten()(d5)
        d6 = Dense(1)(d6)
    
        model = Model(inputs=x, outputs=d6)
    
        return model

    def dis_loss(self, real_output, fake_output):

        d_loss = tf.reduce_mean(fake_output - real_output)
        
        return d_loss



    def gen_loss(self, fake_output):
        g_loss = -tf.reduce_mean(fake_output)

        return g_loss

    @tf.function
    def train_dis(self, z_shape=None, real_images=None):
        
        z = tf.random.normal(z_shape)

        with tf.GradientTape() as dis_tape:
            generated_images = self.Gz(z)

            fake_output = self.Dz(generated_images)
            real_output = self.Dz(real_images)
            d_loss = self.dis_loss(real_output, fake_output)
            d_pen_loss = self.d_penalty(partial(self.Dz, training=True), real_images, generated_images)
            d_total_loss = d_loss + d_pen_loss

        d_gradients = dis_tape.gradient(d_total_loss, self.Dz.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.Dz.trainable_variables))
        
        return d_loss
    
    @tf.function
    def d_penalty(self, fun, real_imgs, fake_imgs):
        epsi = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
        diff = fake_imgs - real_imgs
        inter_imgs = real_imgs + (epsi * diff)

        with tf.GradientTape() as t:
            t.watch(inter_imgs)
            pred = fun(inter_imgs)

        grad = t.gradient(pred, inter_imgs)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.)**2)

        return gp

    @tf.function
    def train_gen(self, z_shape=None):
        z = tf.random.normal(z_shape)
        with tf.GradientTape() as gen_tape:
            generated_images = self.Gz(z)
            
            fake_output = self.Dz(generated_images)
            g_loss = self.gen_loss(fake_output)

        g_gradients = gen_tape.gradient(g_loss, self.Gz.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.Gz.trainable_variables))

        return g_loss

if __name__ == '__main__':
    a = DCGAN()

