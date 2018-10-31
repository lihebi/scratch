from __future__ import print_function
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


def tensor_basic():
    # creating tensors
    torch.empty(5,3)
    torch.rand(5,3)
    x = torch.tensor([5.5, 3])
    x.size()
    # modify x
    x.new_ones(5, 3, dtype=torch.double)
    torch.randn_like(x, dtype=torch.float)
    y1 = torch.rand(5,3)
    y2 = torch.rand(5,3)
    y1 + y2
    torch.add(y1, y2)

    # GPU
    torch.cuda.is_available()

    if torch.cuda.is_available():
        # this device is used to move tensors in and out of GPU
        device = torch.device("cuda")
        # directly create a tensor on GPU
        y = torch.ones_like(x, device=device)
        # or just use strings ``.to("cuda")``
        x = x.to(device)
        z = x + y
        print(z)
        # ``.to`` can also change dtype together!
        print(z.to("cpu", torch.double))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square
        # convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net3(nn.Module):
    """This class takes 3-channel images
    """
    def __init__(self):
        super(Net3, self).__init__()
        # this "3" is channel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def neural_net_example():
    ##############################
    ## Define the Neural Nets
    ##############################
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    # this input seems to be init?
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    net.zero_grad()
    out.backward(torch.randn(1, 10))

    ##############################
    ## Compute the loss
    ##############################
    output = net(input)
    # we were using random dummy input, now use a dummy target
    target = torch.randn(10)
    target = target.view(1, -1)  # make it the same shape as output
    # use MSE loss
    criterion = nn.MSELoss()
    # compute the loss
    loss = criterion(output, target)
    print(loss)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


    net.zero_grad()     # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    # this is 00000
    print(net.conv1.bias.grad)

    # we are using loss function to control the computations of
    # backprop (technically the update of gradients??)
    loss.backward()

    print('conv1.bias.grad after backward')
    # now contains weights (not weight!)
    print(net.conv1.bias.grad)


    ##############################
    ## update the weights
    ##############################
    # weight update rule: SGD
    # weight = weight - learning_rate * gradient
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    # create your optimizer
    # You can use different optimizers (update rules), other than SGD
    # e.g. Nesterov-SGD, Adam, RMSProp
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

def training():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4,
        shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4,
        shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net3()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        # Oh, trainloader, so fancy
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # so easy?
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            #
            # add up running loss? Yes, because we are doing 2000
            # mini-batches.
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                # So we are reseting the loss here. But how the loss
                # keeps going down? Because we have the `net`! It is
                # updated, and the outputs are computed by applying
                # the net onto the input. The loss is of this output.
                running_loss = 0.0

    print('Finished Training')

    # Finally let's test the network!
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # Test on some image
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ',
          ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # So, this is the output
    outputs = net(images)
    # the output is actually the likelihood of each of the 10 classes,
    # thus we pick the max.
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    # test on all images, calculate accuracy (55%) ???
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # see results by class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    ##############################
    # TODO train on GPU
    ##############################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # transfer net to device
    net.to(device)
    # transfer inputs to device ??? How does this work, exactly?
    inputs, labels = inputs.to(device), labels.to(device)
    # Probably, the net and inputs, labels are all on device. Then,
    # just simply execute previous code will make the computation on
    # GPU.


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # FIXME seems that plt.imshow will only send image for showing,
    # but the window is not displayed. Calling plt.show() will fire it
    # up. Thus the following is added by me.
    plt.show()

def matplotlib_test():
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("test.png")
    # plt.show()

if __name__ == '__main__':
    tensor_basic()

