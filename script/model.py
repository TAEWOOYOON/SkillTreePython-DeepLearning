import tensorflow as tf
from tensorflow.keras import models, layers, utils


class BuildModel() :
  def __init__(self, img_shape, z_dim) :
    self.img_shape = img_shape
    self.z_dim = z_dim
# Generator

  def build_generator(self,kernel_size = 5, activation = 'relu', last_activation = 'sigmoid') :
    h,w,c = self.img_shape
    z = layers.Input(shape = [self.z_dim]) # 길이가 z_dim인 벡터
    y = layers.Dense(int(w/4)*int(h/4)*128)(z)
    y = layers.Reshape([int(w/4),int(h/4),128])(y)

    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(64,kernel_size = kernel_size, padding = 'same', strides = 2, activation = activation)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(c,kernel_size = kernel_size, padding = 'same', strides = 2, activation = last_activation)(y)
    return models.Model(z, y, name = 'Generator')


  # Discriminator

  def build_discriminator(self, kernel_size = 5, activation = 'relu', last_activation = 'sigmoid') :
    x = layers.Input(shape = self.img_shape)
    y = layers.Conv2D(64,kernel_size = kernel_size, strides = 2, padding = 'same', activation = activation)(x)
    y = layers.Dropout(.5)(y)
    y = layers.Conv2D(128,kernel_size = kernel_size, strides = 2, padding = 'same', activation = activation)(y)
    y = layers.Dropout(.5)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(1, activation = last_activation)(y)
    return models.Model(x, y, name = 'Discriminator')