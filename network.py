"""
network.py
~~~~~~~~~~
用SGD+BP来训练一个简单网络

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    # sizes为一个数组，例如[2,3,1]，表示输入层2个神经元，隐层3个神经元，输出层1个神经元
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        # 层数
        self.num_layers = len(sizes)
        # 形状，例如[2,3,1]
        self.sizes = sizes
        # 用np生成均值为0，标准差为1的正态分布随机数
        # biases是一个数组，元素为向量，
        # 第一个元素是3行1列（本层3个神经元就是3行，偏置永远是1列）的随机数组，第二个元素的1行1列的随机数组
        # weights是一个数组，元素为矩阵（形状是上一层的个数*本层个数），
        # 第一个元素是3行2列（本层是3个神经元，就是3行，前一层2个神经元就是2列）的随机数组，第二个元素是1行3列的随机数组
        # 注意net.weights[0]表示隐层的权重矩阵，net.weights[1]表示输出层的权重矩阵，
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # 输入样本a，前馈方法将给出网络的输出值
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            # 第一次时，传入的a为输入向量，计算出第一个隐层的输出，
            # 之后每一次，右侧的a均为上一层的激活值，左侧的a为本层激活值，
            # 输出层的激活值，正是最终的输出值，多么优雅！
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            # 通过 mini_batch_size，相当于以等差数列的形式，切出n/mini_batch_size个mini-batch，组成mini_batches数组
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # 遍历所有mini_batch，即为一个epoch
            for mini_batch in mini_batches:
                # 使用mini_batch里的若干样本，以及学习率eta，对权重和偏置进行一次迭代。
                self.update_mini_batch(mini_batch, eta)
            # 测试，该步为可选；如果提供了测试集，将输出本轮epoch的测试结果
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    # 使用mini_batch里的若干样本和学习率eta，完成一次权重的更新
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # nabla表示倒三角算子，即梯度
        # nabla_w为数组，长度为层数-1，元素是矩阵，对应每一层的权重矩阵，其形状为本层神经元个数*上层神经元个数
        # nabla_w为数组，长度为层数-1，元素为向量，对应每一层的偏置向量，其形状为本层神经元个数*1
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # x表示输入，y表示目标值。
            # 获得单个样本x，y时，代价对w和b的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 太牛了这一步！！！对于每个mini-batch
            # 第一个样本x,y进来后，右侧的nabla_b为零向量数组，长度为层数，[全零的列向量1,全零的列向量2]
            # 此时delta_nabla_b为第一个样本计算出来的偏置梯度数组，与nabla_b有相同的长度、相同的元素尺寸！
            # 这样便计算出了第一个样本对应的梯度向量的数组，形状跟这两个数组的尺寸、元素尺寸一模一样！
            # 第二个样本x,y进来后，右侧的nabla_b为上一个样本计算出的梯度向量数组，与本轮的梯度向量数组，对应的向量按元素相加！
            # 直到最后一个样本，获得的nabla_b将是整个mini_batch上计算出的累加梯度总和，依然为一个数组，跟上面的所有数据拥有相同的尺寸、元素尺寸！
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # nabla_w跟上面同理，也是一个数组，长度是层数-1，只是元素换成了矩阵，每个矩阵的尺寸为本层神经元个数*上层神经元个数
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # nabla_w是梯度向量，nw为其元素
        # w是矩阵，nw也是矩阵，形状均为本层个数*上层个数
        # nw是矩阵，尺寸同w，nw是在整个mini_batch上，每个元素均为累加每一个样本代价对该位置的权重的误差（代价对本）求出来的总和。
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        # nabla_b是数组，长度是层数-1，元素为零向量
        # nabla_w是数组，长度是层数-1，元素为零矩阵，尺寸为本层神经元个数*上层神经元个数
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        # 输入层的激活值
        activation = x
        # 各层的激活值
        activations = [x] # list to store all the activations, layer by layer
        # 储存各层的加权和
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # z为加权和，w是矩阵 本层个数*上层个数  activation为激活值向量 上层个数*1,    b为向量 本层个数*1
            # 因此z为一个向量，本层个数*1
            z = np.dot(w, activation)+b
            zs.append(z)
            # 计算激活值，并依次存入activations
            activation = sigmoid(z)
            activations.append(activation)

        # 计算误差
        # backward pass
        # cost_derivative是计算代价对输出层激活值的梯度，返回一个列向量，行数为输出层神经元个数
        # sigmoid_prime是计算本层激活函数的导数，zs[-1]表示本层（即输出层）的输入加权和。也是一个列向量。
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # delta即是输出层偏置的误差向量
        nabla_b[-1] = delta
        # 而要计算权重的误差矩阵，还需要给delta乘以上一层的激活值      本层个数*1   1*上层个数
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        # 传播这个误差
        # num_layers是层数
        for l in range(2, self.num_layers):
            # 第一次进来，z是倒数第2层的加权和，以后依次是倒数第3、4、。。。遍历下去。
            z = zs[-l]
            # 计算获得本层的激活函数的导数sp，是个向量，行数为本层个数
            sp = sigmoid_prime(z)
            # 后一层的权重矩阵转置*后一层的误差   本层个数*后一层个数  后一层个数*1  结果为本层个数*1 然后两个向量进行内积获得本层的delta
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # 储存本层的delta为偏置的梯度向量
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # 最终获得的nabla_b是一个数组，长度为层数-1，元素为向量，本层个数*1，对应代价对本层偏置的导数（本层误差）
        # 最终获得的nabla_w是一个数组，长度为层数-1，元素为矩阵，本层个数*上层个数，对应代价对本层权重的导数（本层误差*上一层激活值）
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    # 注意：这里的z是一个向量，np会自动按元素进行逐个计算
    return 1.0/(1.0+np.exp(-z))

# 用于反向传播中计算激活函数的导数
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
