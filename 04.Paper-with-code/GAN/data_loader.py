import tensorflow as tf
import numpy as np

# 정규화 - minmax : 0~1, standard norm : -1~1
def mnist_loader(standard = False) :
  (x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
  x_train = x_train / 255. # 0~1
  x_test = x_test / 255. # 0~1
  if standard :
    x_train = (x_train*2)-1
    x_test = (x_test*2)-1

  return x_train,x_test,y_train,y_test

def fmnist_loader(standard = False) :
  (x_train, y_train), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
  x_train = x_train / 255.
  x_test = x_test / 255.
  if standard :
    x_train = (x_train*2)-1
    x_test = (x_test*2)-1

  return x_train,x_test,y_train,y_test

def cifar10_loader(standard = False) :
  (x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train / 255.
  x_test = x_test / 255.
  if standard :
    x_train = (x_train*2)-1
    x_test = (x_test*2)-1

  return x_train,x_test,y_train,y_test