import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#initailize learning rate and training cycles (epochs)
lr = 0.01
epochs = 200

#create storage arrays for weights and biases
weights = np.ones(epochs)
biases = np.ones(epochs)

class linRegModel(object):
    def __init__(self):
        #initialize weight and bias
        self.w = np.array([0], dtype = float)
        self.b = np.array([0], dtype = float)

    def predict(self, x):
        #y = wx +b
        return x * self.w + self.b

    def train(self, x, y):
        for i in range(epochs):
            #calculate errors and derivatives of errors
            error = self.predict(x) - y
            errorPrime_w = 2*(self.predict(x) - y)*x
            errorPrime_b = 2*(self.predict(x) - y)

            #calculate adjustmets for w and b
            adjustment_w = -lr*errorPrime_w * 0.05
            adjustment_b = -lr*errorPrime_b

            #add w and b to storage arrays
            weights[i] = self.w
            biases[i] = self.b

            #adjust w and b
            self.w += adjustment_w.mean()
            self.b += adjustment_b.mean()

            #print error every 20 epochs
            if i % 20 == 0:
                print('Error: ' + str(np.abs(error).mean()))

#create random dataset
data_size = 400
X = np.linspace(0, 10, data_size)
Y = 5*np.random.random(data_size)+X*2 + 5

#initialize and train model
model = linRegModel()
model.train(X, Y)

#initialize plotting data
fig = plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(xlim=(0, 10.2), ylim=(0, 30))
line, = ax.plot([], [], lw=2)

#modifies animation speed
k = 3

#initialize line
def init():
    line.set_data([], [])
    return line,

#animate line
def animate(i):
    y = weights[(i-1)*k]*X + biases[(i-1)*k]
    line.set_data(X, y)
    return line,

#create animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(epochs/k), interval=50, blit=True)

#display random data
plt.scatter(X,Y, c = 'r', alpha = 0.25)

#save and show animation
anim.save('LinRegr.gif', writer='imagemagick')
plt.show()
