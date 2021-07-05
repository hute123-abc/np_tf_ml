import tensorflow as tf
import numpy as np


def conv_op(x, n_out, training, useBN, kh=7, kw=7, dh=1, dw=1, padding="SAME", activation=tf.nn.relu,
            ):
    n_in = x.get_shape()[-1].value

    w = tf.Variable(tf.zeros([kh, kw, n_in, n_out]) + 0.9999)
    b = tf.Variable(tf.zeros([n_out]) + 0.9999)
    #w = tf.get_variable("w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
    #                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #b = tf.get_variable("b", shape=[n_out], dtype=tf.float32,
    #                    initializer=tf.constant_initializer(0.01))
    conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
    z = tf.nn.bias_add(conv, b)

    if useBN:
        z = tf.layers.batch_normalization(z, trainable=training)
    if activation:
        z = activation(z)

    tf.summary.histogram('conv_op' + '/outputs', z)
    conv_grads_w = tf.gradients(z, w)
    conv_grads_b = tf.gradients(z, b)
    return z, w, b, conv_grads_w, conv_grads_b


def max_pool_op(x, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    pool_layer_out = tf.nn.max_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding)
    pool_grads = tf.gradients(pool_layer_out, x)
    return pool_layer_out, pool_grads


def avg_pool_op(x, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    pool_layer_out = tf.nn.avg_pool(x,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding)
    pool_grads = tf.gradients(pool_layer_out, x)
    return pool_layer_out, pool_grads


def fc_op(x, n_out, activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value

    w = tf.get_variable("w", shape=[n_in, n_out],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.9999))
    b = tf.get_variable("b", shape=[n_out], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.9999))

    fc = tf.matmul(x, w) + b

    out = fc
    #out = activation(fc)

    fc_grads_w = tf.gradients(out, w)
    fc_grads_b = tf.gradients(out, b)

    return fc, w, out, fc_grads_w, fc_grads_b


def cost(logits, labels):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    cross_entropy_cost = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('loss', cross_entropy_cost)
    loss_grads = tf.gradients(cross_entropy_cost, logits)
    return cross_entropy, cross_entropy_cost, loss_grads


tf.reset_default_graph()

#X_train = np.random.rand(128, 24, 24, 3)
#y_train = np.random.randint(0, 2, size=(128, 10))

X_train = np.load('x_input.npy')
y_train = np.load('y_input.npy')


keep_prob = tf.placeholder(tf.float32)
x_input = tf.placeholder(tf.float32, [128, 24, 24, 3])
y_input = tf.placeholder(tf.float32, [128, 10])


conv1, w, b, conv_grad_w, conv_grad_b = conv_op(x_input, 32, True, False, 7, 7, 1, 1)
#pool2, pool_grad = max_pool_op(conv1, kh=7, kw=7, dh=1, dw=1, padding="SAME")
shape = conv1.get_shape()
fc_in = tf.reshape(conv1, [-1, shape[1].value * shape[2].value * shape[3].value])
logits, w2, prob, fc_grads_w, fc_grads_b = fc_op(fc_in, 10, activation=tf.nn.softmax)


cross_entropy, cross_entropy_cost, loss_grads = cost(labels=y_input, logits=logits)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)


grads = tf.gradients(cross_entropy_cost, tf.trainable_variables())
grads_v = []


for grad, var in list(zip(grads, tf.trainable_variables())):
    tf.summary.histogram(var.name + '/gradient', grad)


sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs2/train", sess.graph)
test_writer = tf.summary.FileWriter("logs2/test", sess.graph)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1):
        # here to determine the keeping probability
        _, o_b, o_w2, o_w, o_conv1, o_grad, o_prob, o_fc_in, o_cross_entropy, loss, o_conv_grad_w, o_conv_grad_b, o_fc_grads_w, o_fc_grads_b, o_loss_grads = sess.run([train_step, b, w2, w, conv1, grads, prob, fc_in, cross_entropy, cross_entropy_cost, conv_grad_w, conv_grad_b, fc_grads_w, fc_grads_b, loss_grads], feed_dict={x_input: X_train, y_input: y_train, keep_prob: 1})
        #loss = sess.run(cross_entropy_cost, feed_dict={x_input: X_train, y_input: y_train, keep_prob: 1})
        print("loss", loss)
        train_result = sess.run(merged, feed_dict={x_input: X_train, y_input: y_train, keep_prob: 1})
        train_writer.add_summary(train_result, i)


