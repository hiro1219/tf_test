# -*- coding:utf-8 -*-

# ライブラリのインポート
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# データセットの取得
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# モデルの設定
# x' = sigmoid(w_2*sigmoid(w_1*x+b_1)+b2)というモデル
x = tf.placeholder(tf.float32, [None, 784])

w_1 = tf.Variable(tf.random_normal([784, 625], mean=0.0, stddev=0.5), dtype=tf.float32)
b_1 = tf.Variable(tf.zeros([625]), dtype=tf.float32)
h_1 = tf.sigmoid(tf.matmul(x, w_1) + b_1)
w_2 = tf.Variable(tf.random_normal([625, 784], mean=0.0, stddev=0.5), dtype=tf.float32)
b_2 = tf.Variable(tf.zeros([784]), dtype=tf.float32)
x_dash = tf.sigmoid(tf.matmul(h_1, w_2) + b_2)

# lossの設定
# クロスエントロピー sigma -x*log(x')-(1-x)*log(1-x')
# log(x)のxが0になるとエラーが起きるので小さな値(10^(-10))を足してそれを防ぐ
cross_entropy = -1. * x * tf.log(x_dash + 1e-10) - (1. - x) * tf.log(1. - x_dash + 1e-10)
loss = tf.reduce_sum(cross_entropy)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# おまじない(初期化)とbatch_size, epoch数の設定
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 100
epoch = 1000

# 学習
for i in range(epoch):
    step = i + 1
    batch_x, batch_y = mnist.train.next_batch(batch_size) # バッチからデータ生成
    sess.run(train, feed_dict={x:batch_x}) # 学習

    # 時々lossと学習結果を出力
    # テストデータの最初の10個の画像の復元をしてみる
    x_test = mnist.test.images[0:10]
    if step % 100 == 0 or step == 1:
        tmp_loss = sess.run(loss, feed_dict={x:batch_x})
        print("step:%d, loss:%6f" % (step, tmp_loss))
        plt.figure(figsize=(20, 4))
        for i in range(10):
            ax = plt.subplot(2, 10, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, 10, i + 11)
            output_imgs = sess.run(x_dash, feed_dict={x:x_test})
            plt.imshow(output_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        file_name = "figure_step_" + str(step) + ".png"
        plt.savefig(file_name)
