import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable, grad
import torch.nn.init as init


random.seed(3)

def rand_cluster(n,c,r):
    """returns n random points in disk of radius r centered at c"""
    x,y = c
    points = []
    for i in range(n):
        theta = 2*math.pi*random.random()
        s = r*random.random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
    return points

def rand_clusters(k,n,r, a,b,c,d):
    """return k clusters of n points each in random disks of radius r
where the centers of the disk are chosen randomly in [a,b]x[c,d]"""
    clusters = []
    for _ in range(k):
        x = a + (b-a)*random.random()
        y = c + (d-c)*random.random()
        clusters.extend(rand_cluster(n,(x,y),r))
    return clusters

n = 50
X = rand_clusters(2,50,0.8,-1,1,-1,1)
X = np.array(X)
# y = np.array([[1]*n + [0]*n, [0]*n + [1]*n])
y = np.array([1]*n + [0]*n)
# print (X, y)

# plt.scatter(X[:n,0], X[:n,1], color=['red'])
# plt.scatter(X[n:,0], X[n:,1], color=['green'])
# plt.show()


class fnn(torch.nn.Module):
    def __init__(self, input_dim = 2, num_classes = 2, n_hidden_neuron = 3, bias = True, lmd = 0.01):
        super(fnn, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, n_hidden_neuron, bias=bias)
        self.rl = torch.nn.Sigmoid()
        self.l2 = torch.nn.Linear(n_hidden_neuron, num_classes, bias=bias)
        # self.params = [self.linear.weight, self.linear.bias]
        # self.params = [self.linear.weight]
        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.CrossEntropyLoss()
        self.lmd = lmd

    def init(self):
        init.xavier_uniform(self.l1.weight, gain=np.sqrt(0.8))
        init.xavier_uniform(self.l2.weight, gain=np.sqrt(0.8))
        # init.constant(self.l1.bias, 0.)
        # init.constant(self.l2.bias, 0.)

    def forward(self, x):

        x1 = self.l1(x)
        x11 = self.rl(x1)
        x2 = self.l2(x11)

        return x2

    def get_loss(self, x, y):
    
        pred = self.forward(x)
        loss = self.loss(pred, y)
        return loss

    def get_grad(self, x, y):

        out = self.get_loss(x, y)
        grad = torch.autograd.grad(out, self.parameters(), create_graph=True)
        i = 0
        for p in model.parameters():
            grad[i].data.add_(self.lmd * p.data)
            i += 1
        return grad

    def get_grad_norm(self, x, y):
        grad = self.get_grad(x, y)
        norm = 0.
        for i in grad:
            norm += torch.norm(i).data.numpy()[0]**2
        return norm ** 0.5

    def get_pred_acc(self, x, y):
        pred = self.forward(x)
        pred_y = torch.max(pred, 1)[1].data.squeeze()
        accuracy = sum(pred_y == y.data.squeeze())

        return 1. * accuracy / y.size()[0]


X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
# y = Variable(torch.from_numpy(y).type(torch.FloatTensor))
y = Variable(torch.from_numpy(y).type(torch.LongTensor))

lmd = 0.01

model = fnn(n_hidden_neuron=3, bias=False, lmd=lmd)
model.init()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=lmd)
for t in range(500):
  # Forward pass: Compute predicted y by passing x to the model

    # Compute and print loss
    loss = model.get_loss(X, y)
    # print(t, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

norm_grad = model.get_grad_norm(X, y)
acc = model.get_pred_acc(X,y)
print (loss.data[0], norm_grad, acc)
print (model.l1.weight.data.numpy())
print (model.l2.weight.data.numpy())