from pydantic import BaseModel
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"

class DataConfig(BaseModel):
    raw_input_dir: str
    raw_input_file: str

class TestConfig(BaseModel):
    output_dir: str

class ExperimentConfig(BaseModel):
    outputs_dir: str
    configs_dir: str
    scripts_dir: str

class RuntimeConfig(BaseModel):
    output_dir: str

class GlobalConfig(BaseModel):
    data: DataConfig
    test: TestConfig
    experiment: ExperimentConfig
    runtime: RuntimeConfig

def get_cached_global_config(path: Path | None = None) -> GlobalConfig:
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return GlobalConfig.model_validate(data)

CONFIG = get_cached_global_config(CONFIG_PATH)
