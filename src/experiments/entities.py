from dataclasses import dataclass


@dataclass
class ExpGroupConfig_VelCoeffs:
    # ranges are (min, max, step)
    inertia_range: tuple[float, float, float]
    personal_range: tuple[float, float, float]
    global_range: tuple[float, float, float]
    social_range: tuple[float, float, float]

@dataclass
class ExpConfig_VelCoeffs:
    inertia: float
    personal: float
    global_: float
    social: float

@dataclass
class ExpGroupConfig_FitnessBudget:
    fitness_budget: int
    swarm_size_range: tuple[int, int, int] # ranges are (min, max, step)

@dataclass
class ExpConfig_FitnessBudget:
    swarm_size: int
    iterations_count: int