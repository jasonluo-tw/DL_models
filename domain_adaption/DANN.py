import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
import numpy as np

@tf.custom_gradient
def GradientReverseOp(x):
    def grad(dy):
        return -1 * dy
    return x, grad

## GradientReverseLayer
class GradientReverseLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def call(self, inputs):
        return GradientReverseOp(inputs)

class DANN():
    def __init__(self, input_shape):

        self.fea_extractor = Sequential([
            Input(shape=input_shape),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
        ])

        self.value_predictor = Sequential([
            Dense(128, activation='relu'),
            #Dropout(0.3),
            Dense(1, activation='linear')
        ])
        ## domain classifier
        self.domain_predictor = Sequential([
            GradientReverseLayer(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid'),
        ])

        self.pred_value = Sequential([
            self.fea_extractor,
            self.value_predictor
        ])

        self.pred_domain = Sequential([
            self.fea_extractor,
            self.domain_predictor
        ])

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.binary_loss = tf.keras.losses.BinaryCrossentropy()

        self.lp_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.dc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.fe_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train(self, x_source, x_target, y_source):
        label1 = tf.zeros((len(x_source)), 1)
        label2 = tf.ones((len(x_target)), 1)
        domain_labels = tf.concat((label1, label2), axis=0)

        all_x = tf.concat((x_source, x_target), axis=0)
        
        ## predict value (regression)
        with tf.GradientTape() as tape:
            y_value_pred = self.pred_value(x_source, training=True)
            mse_loss = self.mse_loss(y_source, y_value_pred)
        lp_grad = tape.gradient(mse_loss, self.pred_value.trainable_variables)

        ## classify domain (including reverse gradient)
        with tf.GradientTape(persistent=True) as tape:
            y_domain_pred = self.pred_domain(all_x, training=True)
            dc_loss = self.binary_loss(domain_labels, y_domain_pred)
        fe_grad = tape.gradient(dc_loss, self.fea_extractor.trainable_variables)
        dp_grad = tape.gradient(dc_loss, self.domain_predictor.trainable_variables)
        ## Because using persistent=True to calculate the gradient again, 
        ## the tape has to be deleted manually
        del tape

        self.lp_optimizer.apply_gradients(zip(lp_grad, self.pred_value.trainable_variables))
        self.dc_optimizer.apply_gradients(zip(dp_grad, self.domain_predictor.trainable_variables))
        self.fe_optimizer.apply_gradients(zip(fe_grad, self.fea_extractor.trainable_variables))

        return mse_loss, dc_loss

    @tf.function
    def train_source(self, x_source, y_source):
        with tf.GradientTape() as tape:
            y_value_pred = self.pred_value(x_source, training=True)
            mse_loss = self.mse_loss(y_source, y_value_pred)
        grad = tape.gradient(mse_loss, self.pred_value.trainable_variables)

        self.lp_optimizer.apply_gradients(zip(grad, self.pred_value.trainable_variables))

        return mse_loss

    def evaluate(self, x, y):
        pred_y = np.squeeze(self.pred_value(x).numpy())
        rmse = np.sqrt(np.mean( (pred_y-y)**2 ))

        return rmse

if __name__ == '__main__':
    import numpy as np
    model = DANN((16))
    trains = np.random.random(size=(120, 16))
    labels = np.random.random(size=(120, 1))
    
    target_ = np.random.random(size=(100, 16))

    mse_loss, dc_loss = model.train(trains, target_, labels)
    print(mse_loss, dc_loss)
    rmse = model.evaluate(trains, labels)
    print(rmse)
    #print(model.pred_value(np.zeros((1, 16))))
