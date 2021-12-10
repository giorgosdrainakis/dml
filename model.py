import torch.nn as nn
import torch.nn.functional as F
import Tools
import Global

class MNIST_Model(nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()
        self.fc1 = nn.Linear(784, 500) #28x28 images
        self.fc2 = nn.Linear(500, 10)
        self.size=float(1.55) #Mb

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_size(self):
        return self.size

class INFIMNIST_Model_big(nn.Module):
    def __init__(self):
        super(INFIMNIST_Model_big, self).__init__()
        self.fc1 = nn.Linear(784, 512) #28x28 images
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.size=float(1.55) #Mb

    def forward(self, xb):
        out = xb.view(xb.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size

class INFIMNIST_Model(nn.Module):
    def __init__(self):
        super(INFIMNIST_Model, self).__init__()
        self.fc1 = nn.Linear(784, 256) #28x28 images
        self.fc2 = nn.Linear(256, 10)
        self.size=float(0.8) #Mb

    def forward(self, xb):
        out = xb.view(-1,784)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size

class SVHN_Model_big(nn.Module):
    def __init__(self):
        super(SVHN_Model_big, self).__init__()
        self.fc1 = nn.Linear(3072, 2048) #32x32 images
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 10)
        self.size=float(35.5) #Mb todo

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        out = F.relu(out)
        out = self.fc6(out)
        out = F.relu(out)
        out = self.fc7(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size

class SVHN_Model(nn.Module):
    def __init__(self):
        super(SVHN_Model, self).__init__()
        self.fc3 = nn.Linear(3072, 512) #32x32x3
        self.fc5 = nn.Linear(512, 10)
        self.size=float(6.1) #Mb todo

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc5(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size

    def get_train_time_mobile_with_epochs(self,samples,epochs):
        return float(epochs*(samples/125))

class SVHN_Model_6(nn.Module):
    def __init__(self):
        super(SVHN_Model_6, self).__init__()
        self.fc3 = nn.Linear(3072, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 10)
        self.size=float(6.8) #Mb todo

    def forward(self, xb):
        out = xb.view(-1, 3072)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        out = F.relu(out)
        out = self.fc6(out)
        out = F.relu(out)
        out = self.fc7(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size

class CIFAR10_Model(nn.Module):
    def __init__(self):
        super(CIFAR10_Model, self).__init__()
        self.fc1 = nn.Linear(3072, 512) #32x32 images
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.size=float(6.8) #Mb

    def forward(self, xb):
        out = xb.view(-1, 3072)
        #out = xb.view(xb.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return F.log_softmax(out, dim=1)

    def get_size(self):
        return self.size