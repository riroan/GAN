import keras
from keras import layers
from keras import initializers

latent_dim = 100
channels = 3
height = 64
width = 64

class GAN():
    def __init__(self):
        self.gan = None
        self.generator = None
        self.discriminator = None
        
    def generate_GAN(self):
        generator_input = keras.Input(shape=(latent_dim,))
        
        x = layers.Dense(4*4*1024,kernel_initializer=initializers.random_normal(stddev=0.02))(generator_input)
        x = layers.Reshape((4, 4, 1024))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(512, 5, strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(256, 5, strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(128, 5, strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(channels, 5, strides=2, activation='tanh', padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        self.generator = keras.models.Model(generator_input, x)
        self.generator.summary()
        
        discriminator_input = layers.Input(shape=(height, width, channels))
        x = layers.Conv2D(64, 5 ,strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(discriminator_input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(128, 5, strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(256, 5, strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(256, 5, strides=2, padding='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        #x = layers.BatchNormalization(momentum=0.5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(512,5,strides = 2, padding ='same',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid',kernel_initializer=initializers.random_normal(stddev=0.02))(x)
        
        self.discriminator = keras.models.Model(discriminator_input, x)
        self.discriminator.summary()
        
        discriminator_optimizer = keras.optimizers.Adam(lr=0.00005, beta_1=0.5)
        self.discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
        
        self.discriminator.trainable = False
        
        gan_input = keras.Input(shape=(latent_dim,))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = keras.models.Model(gan_input, gan_output)
        
        gan_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2 = 0.999)
        self.gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
        return self.gan
    
    def load(self,path):
        self.gan.load_weights(path)
    
    def save(self,path):
        self.gan.save_weights(path)
        