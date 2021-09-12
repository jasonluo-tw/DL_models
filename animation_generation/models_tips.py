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
        soft_ones = tf.random.uniform(tf.shape(real_output), minval=0.7, maxval=1.2)
        soft_zeros = tf.random.uniform(tf.shape(fake_output), minval=0.0, maxval=0.3)
        
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=soft_ones))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=soft_zeros))
    
        #d_real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        #d_fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)

        d_total_loss = d_real_loss + d_fake_loss

        return d_total_loss

    def gen_loss(self, fake_output):
        soft_ones = tf.random.uniform(tf.shape(fake_output), minval=0.7, maxval=1.2)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=soft_ones))
        
        #g_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        return g_loss

    @tf.function
    def train_dis(self, z_shape=None, real_images=None):
        
        z = tf.random.normal(z_shape)

        with tf.GradientTape() as dis_tape:
            generated_images = self.Gz(z)

            fake_output = self.Dz(generated_images)
            real_output = self.Dz(real_images)
            d_loss = self.dis_loss(real_output, fake_output)
        
        d_gradients = dis_tape.gradient(d_loss, self.Dz.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.Dz.trainable_variables))
        
        return d_loss

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

    @tf.function
    def train_step(self, z_shape=None, real_images=None):
        z = tf.random.normal(z_shape)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_images = self.Gz(z)

            real_output = self.Dz(real_images)
            fake_output = self.Dz(generated_images)

            g_loss = self.gen_loss(fake_output)
            d_loss = self.dis_loss(real_output, fake_output)
        
        g_gradients = gen_tape.gradient(g_loss, self.Gz.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.Gz.trainable_variables))
        
        d_gradients = dis_tape.gradient(d_loss, self.Dz.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.Dz.trainable_variables))

        return g_loss, d_loss

if __name__ == '__main__':
    a = DCGAN()

