from NeuralNetworkTests import TestFullyConnected1, TestReLU, TestSoftMax
from Layers import FullyConnected, ReLU, SoftMax

if __name__ == "__main__":
    # test fully connected
    test_fully_connected = TestFullyConnected1()
    test_fully_connected.setUp()
    test_fully_connected.test_forward_size()
    test_fully_connected.test_backward_size()
    test_fully_connected.testOptimizer()
    test_fully_connected.testOptimizerUpdate()
    test_fully_connected.testOptimizerCalculateUpdate()
    
    # # test ReLU
    # test_relu = TestReLU()
    # test_relu.setUp()
    # test_relu.test_forward()
    # test_relu.test_backward()
    
    # test SoftMax
    # test_softmax = TestSoftMax()
    # test_softmax.setUp()
    # test_softmax.testForward()
    # test_softmax.testBackward()
    
    # test CrossEntropyLoss
    # test_cross_entropy_loss = TestCrossEntropyLoss()
    # test_cross_entropy_loss.setUp()
    # test_cross_entropy_loss.testForward()
    # test_cross_entropy_loss.testBackward()
    # test_cross_entropy_loss.testOptimizer()
    # test_cross_entropy_loss.testOptimizerUpdate()
    # test_cross_entropy_loss.testOptimizerCalculateUpdate()
    
    # print("All tests passed!")
