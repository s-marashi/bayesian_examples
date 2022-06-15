# This notebook will contains examples from chapter 3 of think bayes 2e
#
# _Note: As this notebook is for educational purpose I am using a narrative script._

import numpy as np
import matplotlib.pyplot as plt

# Cookie Problem
# ==============
# Suppose there are two bowld of cookies.
# * Bowl 1 contains 30 vanilla cookies and 10 chocolate cookies.
# * Bowl 2 contains 20 vanilla cookies and 20 chocolate cookies.
#
# Now suppose you choose one of the bowls at random and, without looking, choose a cookie at random. If the cookie is vanilla, what is the probability that it came from Bowl 1?

# Hypothesis: Wether it is choosen from bowl 1 or bowl2
# Prior Probability Distribution: Both of bowls are equally likely
prior_pmf = np.ones(2)
prior_pmf = prior_pmf / prior_pmf.sum()

# Likelihood: As our evidence is a Vanilla cookie we must calculate
# P(Vanilla|Bowl1) and P(Vanilla|Bowl2) as our likliehoods.
likelihood_vanilla = np.array([3/4, 1/2])

# to calculate posterior we must first calculate total probability
p_vanilla = (prior_pmf * likelihood_vanilla).sum()
# then we calculate posterior probabilty
posterior_pmf = prior_pmf * likelihood_vanilla / p_vanilla
posterior_pmf

# now if we put the first cookie back(so likelihoods remains 
# constant) and pick a second cookie again from the same bowl 
# and again it is a vanilla cookie what it pmf for hypothesis
prior_pmf2 = posterior_pmf.copy()
p_vanilla2 = (prior_pmf2 * likelihood_vanilla).sum()
posterior_pmf2 = prior_pmf2 * likelihood_vanilla / p_vanilla2
posterior_pmf2

# now if the third cookie is a chocolate cookie what is the pmf 
# of the hypothesis
prior_pmf3 = posterior_pmf2.copy()
likelihood_chocolate = np.array([0.25, 0.5])
p_chocolate = (prior_pmf3 * likelihood_chocolate).sum()
posterior_pmf3 = prior_pmf3 * likelihood_chocolate / p_chocolate
posterior_pmf3

# 101 Bowls Problem
# ==================
# Next let's solve a cookie problem with 101 bowls:
#
# * Bowl 0 contains 0% vanilla cookies,
#
# * Bowl 1 contains 1% vanilla cookies,
#
# * Bowl 2 contains 2% vanilla cookies,
#
# and so on, up to
#
# * Bowl 99 contains 99% vanilla cookies, and
#
# * Bowl 100 contains all vanilla cookies.
#
# As in the previous version, there are only two kinds of cookies, vanilla and chocolate.  So Bowl 0 is all chocolate cookies, Bowl 1 is 99% chocolate, and so on.
#
# Suppose we choose a bowl at random, choose a cookie at random, and it turns out to be vanilla.  What is the probability that the cookie came from Bowl $x$, for each value of $x$?
#
# To solve this problem, I'll use `np.arange` to make an array that represents 101 hypotheses, numbered from 0 to 100.

# +
# Hypothesis: Probability of bpwl being any bowl between 0 and 100
# and the probability of selection all bowls are equally likely
# so prior pmf is a uniform distribution
prior_pmf = np.ones(101)
prior_pmf = prior_pmf / prior_pmf.sum()

# calculate likelihoods
likelihood_vanilla = np.linspace(0, 1, 101)
likelihood_chocolate = 1 - likelihood_vanilla

# and total probability of both vanilla and chocolate cookies
p_vanilla = (prior_pmf * likelihood_vanilla).sum()
p_chocolate = (prior_pmf * likelihood_chocolate).sum()

posterior_pmf = prior_pmf * likelihood_vanilla / p_vanilla

plt.plot(np.arange(0, 101), posterior_pmf)
# -

# if the second cookie from the same random bowl is a vanilla cookie
prior_pmf2 = posterior_pmf.copy()
p_vanilla2 = (prior_pmf2 * likelihood_vanilla).sum()
posterior_pmf2 = prior_pmf2 * likelihood_vanilla / p_vanilla2
plt.plot(np.arange(0, 101), posterior_pmf2)

# and the third random cookie from the same random bowl 
# is a chocolate cookie
prior_pmf3 = posterior_pmf2.copy()
p_chocolate = (prior_pmf3 * likelihood_chocolate).sum()
posterior_pmf3 = (prior_pmf3 * likelihood_chocolate) / p_chocolate
plt.plot(np.arange(0, 101), posterior_pmf3)

# MAP: Maximum A Posteori probability
# MAP of the posterior distribution is:
posterior_pmf3.argmax()

# The Dice Problem
# ================
# Suppose I have a box with a 6-sided die, an 8-sided die, and a 12-sided die. I choose one of the dice at random, roll it, and report that the outcome is a 1. What is the probability that I chose the 6-sided die?

# since all dice are equally likely prior is uniform
prior_pmf = np.ones(3)
likelihood = np.array([1/6, 1/8, 1/12])
prob = (prior_pmf * likelihood).sum()
posterior = prior_pmf * likelihood / prob
posterior

# if we roll the dice again and we get a 7 posterior will 
# be like this
prior_pmf2 = posterior.copy()
likelihood_7 = np.array([0, 1/8, 1/12])
prob_7 = (prior_pmf2 * likelihood_7).sum()
posterior2 = prior_pmf2 * likelihood_7 / prob_7
posterior2


