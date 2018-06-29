# -*- coding:utf-8 -*-

# ライブラリのインポート
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# データセットの設定
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# モデルの生成
# softmax回帰モデル
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.random_normal([784, 10]), dtype=tf.float32)
b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# クロスエントロピーと学習の設定
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 正解率の計算
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初期化とパラメータの設定
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 100
epoch = 1000

# 学習
for i in range(epoch):
    step = i + 1
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x:batch_x, y_:batch_y})

    if step % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
        tmp_loss = sess.run(cross_entropy, feed_dict={x:batch_x, y_:batch_y})
        print("step:%d, acc:%6f, loss:%6f" % (step, acc, tmp_loss))
