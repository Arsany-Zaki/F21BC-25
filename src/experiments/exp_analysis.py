from dataclasses import dataclass

@dataclass
class ExpAnalysis:
    experiment_name: str
    run_count: int
    best_fitnesses: list[float]
    training_times: list[float]
    