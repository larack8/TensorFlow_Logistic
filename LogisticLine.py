import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

## 多元逻辑回归

# Read x and y
x_data = np.loadtxt("ex4x.dat").astype(np.float32)
y_data = np.loadtxt("ex4y.dat").astype(np.float32)

scaler = preprocessing.StandardScaler().fit(x_data)
x_data_standard = scaler.transform(x_data)

# We evaluate the x and y by sklearn to get a sense of the coefficients.
reg = LogisticRegression(C=999999999, solver="newton-cg")  # Set C as a large positive number to minimize the regularization effect
reg.fit(x_data, y_data)
print ("Coefficients of sklearn: K=%s, b=%f" % (reg.coef_, reg.intercept_))

# Now we use tensorflow to get similar results.
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))
y = 1 / (1 + tf.exp(-tf.matmul(x_data_standard, W) + b))
loss = tf.reduce_mean(- y_data.reshape(-1, 1) *  tf.log(y) - (1 - y_data.reshape(-1, 1)) * tf.log(1 - y))

optimizer = tf.train.GradientDescentOptimizer(1.3)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print (step, sess.run(W).flatten(), sess.run(b).flatten())

print ("Coefficients of tensorflow (input should be standardized): K=%s, b=%s" % (sess.run(W).flatten(), sess.run(b).flatten()))
print ("Coefficients of tensorflow (raw input): K=%s, b=%s" % (sess.run(W).flatten() / scaler.scale_, sess.run(b).flatten() - np.dot(scaler.mean_ / scaler.scale_, sess.run(W))))


# Problem solved and we are happy. But...
# I'd like to implement the logistic regression from a multi-class viewpoint instead of binary.
# In machine learning domain, it is called softmax regression
# In economic and statistics domain, it is called multinomial logit (MNL) model, proposed by Daniel McFadden, who shared the 2000  Nobel Memorial Prize in Economic Sciences.

print ("------------------------------------------------")
print ("We solve this binary classification problem again from the viewpoint of multinomial classification")
print ("------------------------------------------------")

# As a tradition, sklearn first
reg = LogisticRegression(C=9999999999, solver="newton-cg", multi_class="multinomial")
reg.fit(x_data, y_data)
print ("Coefficients of sklearn: K=%s, b=%f" % (reg.coef_, reg.intercept_))
print ("A little bit difference at first glance. What about multiply them with 2?")

# Then try tensorflow
W = tf.Variable(tf.zeros([2, 2]))  # first 2 is feature number, second 2 is class number
b = tf.Variable(tf.zeros([1, 2]))
V = tf.matmul(x_data_standard, W) + b
y = tf.nn.softmax(V)  # tensorflow provide a utility function to calculate the probability of observer n choose alternative i, you can replace it with `y = tf.exp(V) / tf.reduce_sum(tf.exp(V), keep_dims=True, reduction_indices=[1])`

# Encode the y label in one-hot manner
lb = preprocessing.LabelBinarizer()
lb.fit(y_data)
y_data_trans = lb.transform(y_data)
y_data_trans = np.concatenate((1 - y_data_trans, y_data_trans), axis=1)  # Only necessary for binary class

loss = tf.reduce_mean(-tf.reduce_sum(y_data_trans * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(1.3)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print (step, sess.run(W).flatten(), sess.run(b).flatten())

print ("Coefficients of tensorflow (input should be standardized): K=%s, b=%s" % (sess.run(W).flatten(), sess.run(b).flatten()))
print ("Coefficients of tensorflow (raw input): K=%s, b=%s" % ((sess.run(W) / scaler.scale_).flatten(),  sess.run(b).flatten() - np.dot(scaler.mean_ / scaler.scale_, sess.run(W))))