import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

dist = stats.beta
n_trials = [0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 5000]
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0, 1, 100)

for k, N in enumerate(n_trials):
    heads = data[:N].sum()
    y = dist.pdf(x, 1 + heads, 1 + N - heads)
    sx = plt.subplot(len(n_trials) // 2, 2, k+1)
    plt.setp(sx.get_yticklabels(), visible=False)
    plt.plot(x, y, label="observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color="#348ABD", alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)

plt.suptitle("Bayesian updating of posterior probabilities", y=1.0, fontsize=14)
plt.tight_layout()    
plt.show()
