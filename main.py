import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


# # net = network.Network([784, 30, 10])
# net = network.Network([784, 30, 10])
# # (training_data, epochs, mini_batch_size, eta,test_data=None)
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5,
#         evaluation_data=test_data,
#         monitor_evaluation_cost=True,
#         monitor_evaluation_accuracy=True,
#         monitor_training_cost=True,
#         monitor_training_accuracy=True,
#         )

# # 过拟合 正则化
# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5,
#         evaluation_data=test_data,
#         monitor_evaluation_accuracy=True,
#         monitor_training_cost=True)

# # 测试正则化的效果
# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data[:1000], 400, 10, 0.5,
#         evaluation_data=test_data,
#         lmbda = 0.1, # 正则因子
#         monitor_evaluation_cost=True,
#         monitor_evaluation_accuracy=True,
#         monitor_training_cost=True,
#         monitor_training_accuracy=True)


# # 正则化，测试5000个训练样本后，训练正确率与测试正确率的差距
# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5,
#         evaluation_data=test_data, lmbda = 5.0,
#         monitor_evaluation_accuracy=True, monitor_training_accuracy=True)


# 正则化，使用100个隐层单元，学习率0.1，epoch为60，将达到98.04的正确率
import network2
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 60, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)
