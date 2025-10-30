# Experiment and RunSet classes for PSO benchmarking
import numpy as np
from tabulate import tabulate
from pso.pso import PSO
from pso.pso_config import PSOConfig
from pso.pyswarm_pso import PySwarmPSO
from settings.enumerations import BoundaryHandling, InformantSelection

class RunSet:
    def __init__(self, label, pso_class, config, func, n_runs=10):
        self.label = label
        self.pso_class = pso_class
        self.config = config
        self.func = func
        self.n_runs = n_runs
        self.results = []

    def run(self):
        self.results = []
        for _ in range(self.n_runs):
            if self.pso_class == PSO:
                pso = PSO(self.config)
                _, best_fit = pso.optimize(self.func)
            else:
                pso = self.pso_class(self.config)
                _, best_fit = pso.optimize_with_given_config(self.func)
            self.results.append(best_fit)
        self.results = np.array(self.results)
        return self.results

    def mean(self):
        return np.mean(self.results)

    def std(self):
        return np.std(self.results)

class Experiment:
    def __init__(self, label, config, func):
        self.label = label
        self.config = config
        self.func = func
        self.runsets = []

    def add_runset(self, runset):
        self.runsets.append(runset)

    def run(self):
        for runset in self.runsets:
            runset.run()

class ExperimentSet:
    def __init__(self, title):
        self.title = title
        self.experiments = []

    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def run(self):
        print(f"\n=== {self.title} ===")
        for exp in self.experiments:
            exp.run()
        self.print_results()

    def print_results(self):
        headers = ["Experiment"] + [rs.label for rs in self.experiments[0].runsets]
        table = []
        for exp in self.experiments:
            row = [exp.label]
            for rs in exp.runsets:
                row.append(f"mean={rs.mean():.4f}\nstd={rs.std():.4f}")
            table.append(row)
        print(tabulate(table, headers=headers, tablefmt="grid"))
