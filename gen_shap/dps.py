import math
import platypus
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *
import pandas as pd

class CubicDPSLever(Lever):
    def __init__(self, name, length=1, c_bounds=(-2, 2), r_bounds=(0, 2)):
        super().__init__(name)
        self.length = length
        self.c_bounds = c_bounds
        self.r_bounds = r_bounds

    def to_variables(self):
        result = []
        for _ in range(self.length):
            result += [platypus.Real(self.c_bounds[0], self.c_bounds[1])]
            result += [platypus.Real(self.r_bounds[0], self.r_bounds[1])]
            result += [platypus.Real(0, 1)]
        return result

    def from_variables(self, variables):
        policy = {}
        policy["length"] = self.length
        policy["rbfs"] = []

        for i in range(self.length):
            policy["rbfs"] += [{
                "center": variables[i*3+0],
                "radius": variables[i*3+1],
                "weight": variables[i*3+2]
            }]

        weight_sum = sum([p["weight"] for p in policy["rbfs"]])
        for i in range(self.length):
            policy["rbfs"][i]["weight"] /= weight_sum

        return policy

def evaluateCubicDPS(policy, current_value):
    value = 0
    for i in range(policy["length"]):
        rbf = policy["rbfs"][i]
        value += rbf["weight"] * abs((current_value - rbf["center"]) / rbf["radius"])**3
    return min(max(value, 0.01), 0.1)

def lake_problem(policy, b=0.42, q=2.0, mean=0.02, stdev=0.001, alpha=0.4, delta=0.98, nsamples=100, steps=100):
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    X = np.zeros((steps,))
    decisions = np.zeros((steps,))
    average_daily_P = np.zeros((steps,))
    reliability = 0.0
    utility = 0.0
    inertia = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=steps)

        for t in range(1, steps):
            decisions[t-1] = evaluateCubicDPS(policy, X[t-1])
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)

        reliability += np.sum(X < Pcrit)/float(steps)
        utility += np.sum(alpha * decisions * np.power(delta, np.arange(steps)))
        inertia += np.sum(np.diff(decisions) > -0.01)/float(steps-1)

    return (
        np.max(average_daily_P),
        utility / nsamples,
        inertia / nsamples,
        reliability / nsamples
    )

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

model.levers = [CubicDPSLever("policy", length=3)]

setup_cache(file="dps.cache")
output = cache("dps_output", lambda: optimize(model, "NSGAII", 10000))

results = []

for sol in output:
    policy_data = sol["policy"]  # this is the RBF structure returned from CubicDPSLever
    obj = {
        "max_P": sol["max_P"],
        "utility": sol["utility"],
        "inertia": sol["inertia"],
        "reliability": sol["reliability"]
    }

    # Flatten the RBF parameters
    for i, rbf in enumerate(policy_data["rbfs"]):
        obj[f"center_{i}"] = rbf["center"]
        obj[f"radius_{i}"] = rbf["radius"]
        obj[f"weight_{i}"] = rbf["weight"]

    results.append(obj)

df = pd.DataFrame(results)
df.to_csv("../data/pareto_frontier.csv", index=False)

print("Pareto frontier saved as pareto_frontier.csv")

scatter3d(model, output)
plt.show()
