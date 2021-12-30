import numpy as np
import matplotlib.pyplot as plt

from prml.rv import (Bernoulli,
                     Beta)

np.random.seed(1234)
#%%
data = np.array([0, 1,1,1,1])
model = Bernoulli()
model.fit(data)
print("The estimated mu is: ",
      model.mu)
#%%
for i in range(3):
    print("Experiment", i,":",
          model.draw(1))
#%%
x = np.linspace(0,1,100)
for i, [a, b] in enumerate([[0.1, 0.1],
                            [1, 1], [2, 3], [8, 4]]):
    plt.subplot(2, 2, i+1)
    beta = Beta(a, b)
    plt.xlim(0,1)
    plt.ylim(0,3)
    plt.plot(x, beta.pdf(x))
    plt.annotate("a={}".format(a), (0.1, 2.5))
    plt.annotate("b={}".format(b), (0.1, 2.1))

plt.show()
#%%
print("The maximum likelihood estimation")
model = Bernoulli()
model.fit(np.array([1]))
print("{} out of 10000 is 1".format(model.draw(10000).sum()))

print()

print("The Bayesian estimation")
model = Bernoulli(mu=Beta(0.5,1))
model.fit(np.array([1]))
print("{} out of 10000 is 1".format(model.draw(10000).sum()))