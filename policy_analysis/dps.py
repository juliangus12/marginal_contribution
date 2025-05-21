# dps.py - generates Pareto-optimal DPS policies for the shallow lake problem
# author: julian gutierrez, dartmouth college, spring 2025
#
# this module implements the direct policy search (dps) method from:
#   quinn et al. (2017), using rhodium and platypus libraries.
# it defines a cubic radial basis function policy, evaluates it on the lake model,
# and optimizes it via NSGA-II to produce a Pareto front.
#
# outputs: pareto_frontier.csv containing objectives and RBF parameters
# dependencies: rhodium, platypus, numpy, scipy, matplotlib, pandas

import math
import platypus
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *
import pandas as pd

# defines a cubic radial basis function policy with learnable weights
class CubicDPSLever(Lever):
    def __init__(self, name, length=1, c_bounds=(-2, 2), r_bounds=(0, 2)):
        super().__init__(name)
        self.length = length
        self.c_bounds = c_bounds
        self.r_bounds = r_bounds

    # defines parameter encoding for platypus optimizer
    def to_variables(self):
        result = []
        for _ in range(self.length):
            result += [platypus.Real(self.c_bounds[0], self.c_bounds[1])]  # center
            result += [platypus.Real(self.r_bounds[0], self.r_bounds[1])]  # radius
            result += [platypus.Real(0, 1)]  # unnormalized weight
        return result

    # decodes platypus variables into structured policy
    def from_variables(self, variables):
        policy = {"length": self.length, "rbfs": []}
        for i in range(self.length):
            policy["rbfs"].append({
                "center": variables[i*3+0],
                "radius": variables[i*3+1],
                "weight": variables[i*3+2]
            })

        # normalize weights so they sum to 1
        weight_sum = sum(rbf["weight"] for rbf in policy["rbfs"])
        for rbf in policy["rbfs"]:
            rbf["weight"] /= weight_sum

        return policy

# evaluates the policy at a given lake phosphorus level
def evaluateCubicDPS(policy, current_value):
    value = 0
    for rbf in policy["rbfs"]:
        # cubic basis function with absolute value
        value += rbf["weight"] * abs((current_value - rbf["center"]) / rbf["radius"])**3
    return min(max(value, 0.01), 0.1)  # clamp between 0.01 and 0.1

# stochastic simulation of the shallow lake over many inflow samples
def lake_problem(policy, b=0.42, q=2.0, mean=0.02, stdev=0.001, alpha=0.4, delta=0.98, nsamples=100, steps=100):
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)  # eutrophication threshold

    X = np.zeros((steps,))
    decisions = np.zeros((steps,))
    average_daily_P = np.zeros((steps,))
    reliability = utility = inertia = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=steps
        )

        for t in range(1, steps):
            decisions[t-1] = evaluateCubicDPS(policy, X[t-1])
            # phosphorus dynamics with recycling and external inflow
            X[t] = (1 - b) * X[t-1] + X[t-1]**q / (1 + X[t-1]**q) + decisions[t-1] + inflows[t-1]
            average_daily_P[t] += X[t] / float(nsamples)

        reliability += np.sum(X < Pcrit) / steps
        utility += np.sum(alpha * decisions * np.power(delta, np.arange(steps)))
        inertia += np.sum(np.diff(decisions) > -0.01) / (steps - 1)

    return (
        np.max(average_daily_P),
        utility / nsamples,
        inertia / nsamples,
        reliability / nsamples
    )

# define rhodium model using objectives from the quinn et al. paper
model = Model(lake_problem)

model.parameters = [
    Parameter("policy"),
    Parameter("b"),
    Parameter("q"),
    Parameter("mean"),
    Parameter("stdev"),
    Parameter("delta")
]

model.responses = [
    Response("max_P", Response.MINIMIZE),
    Response("utility", Response.MAXIMIZE),
    Response("inertia", Response.MAXIMIZE),
    Response("reliability", Response.MAXIMIZE)
]

# 3-rbf policy (each rbf has center, radius, weight)
model.levers = [CubicDPSLever("policy", length=3)]

# cache optimization results to avoid recomputation
setup_cache(file="dps.cache")
output = cache("dps_output", lambda: optimize(model, "NSGAII", 10000))

# collect objectives + rbf parameters from results
results = []
for sol in output:
    policy_data = sol["policy"]
    obj = {
        "max_P": sol["max_P"],
        "utility": sol["utility"],
        "inertia": sol["inertia"],
        "reliability": sol["reliability"]
    }
    for i, rbf in enumerate(policy_data["rbfs"]):
        obj[f"center_{i}"] = rbf["center"]
        obj[f"radius_{i}"] = rbf["radius"]
        obj[f"weight_{i}"] = rbf["weight"]

    results.append(obj)

# write to csv for downstream analysis
df = pd.DataFrame(results)
df.to_csv("../data/pareto_frontier.csv", index=False)
print("Pareto frontier saved as pareto_frontier.csv")

# visualize policy outcomes in objective space
scatter3d(model, output)
plt.show()
