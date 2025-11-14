from nn.nn import NNConfig
from pso.pso import PSOConfig
from experiments.entities import *

class ExpConfig:
    def __init__(self):
        self.num_exp_runs: int

class ExpRun:
    def __init__(self):
        self.best_fitness: float
        self.time_secs: float

class Exp:
    def __init__(self):
        self.nn_config: NNConfig
        self.pso_config: PSOConfig
        self.exp_runs: list[ExpRun] = []

class ExpGroup_VelCoeffs:
    def __init__(self, config: ExpGroupConfig_VelCoeffs):
        # ranges are (min, max, step)
        self.inertia_range: tuple[float, float, float] = config.inertia_range
        self.personal_range: tuple[float, float, float] = config.personal_range
        self.global_range: tuple[float, float, float] = config.global_range
        self.social_range: tuple[float, float, float] = config.social_range
    
    def _generate_exp_configs(self) -> list[ExpConfig_VelCoeffs]:
        configs = []
        inertia_min, inertia_max, inertia_step = self.inertia_range
        personal_min, personal_max, personal_step = self.personal_range
        global_min, global_max, global_step = self.global_range
        social_min, social_max, social_step = self.social_range

        inertia = inertia_min
        while inertia <= inertia_max:
            personal = personal_min
            while personal <= personal_max:
                global_ = global_min
                while global_ <= global_max:
                    social = social_min
                    while social <= social_max:
                        configs.append(
                            ExpConfig_VelCoeffs(
                                personal=personal, inertia=inertia, global_=global_,social=social))
                        social += social_step
                    global_ += global_step
                personal += personal_step
            inertia += inertia_step
        return configs
    
    def run_exp_group(self):
        pass

class ExpGroup_FitnessBudget:
    def __init__(self, config: ExpGroupConfig_FitnessBudget):
        self.fitness_budget: int = config.fitness_budget
        self.swarm_size_range: tuple[int, int, int] = config.swarm_size_range
    
    def _generate_exp_configs(self) -> list[ExpConfig_FitnessBudget]:
        configs = []
        swarm_size_min, swarm_size_max, swarm_size_step = self.swarm_size_range
        swarm_size = swarm_size_min
        while swarm_size <= swarm_size_max:
            iterations_count = self.fitness_budget // swarm_size
            configs.append(
                ExpConfig_FitnessBudget(
                    swarm_size=swarm_size,
                    iterations_count=iterations_count
                )
            )
            swarm_size += swarm_size_step
        return configs
    
    def run_exp_group(self):
        pass