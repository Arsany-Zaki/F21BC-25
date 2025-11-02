from pydantic import BaseModel
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"

class DataPaths(BaseModel):
    raw_input_dir: str
    raw_input_file: str

class TestPaths(BaseModel):
    output_dir: str

class ExperimentPaths(BaseModel):
    outputs_dir: str
    configs_dir: str
    scripts_dir: str

class RuntimePaths(BaseModel):
    output_dir: str

class Paths(BaseModel):
    data: DataPaths
    test: TestPaths
    experiment: ExperimentPaths
    runtime: RuntimePaths

def _read_paths(path: Path | None = None) -> Paths:
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Paths.model_validate(data)

PATHS = _read_paths(CONFIG_PATH)
