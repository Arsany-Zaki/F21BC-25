import os
import yaml

CONFIG_DIR = "config"
_CONFIG = None         

def _load_all_configs() -> dict:
    merged = {}
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith(".yaml"):
            path = os.path.join(CONFIG_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                merged.update(data)
    return merged

def get_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_all_configs()
    return _CONFIG

CONFIG = get_config()

############ For Testing
if __name__ == "__main__":
  
    cfg = get_config()
    print("dataset_path:", cfg.get("dataset_path"))

    pso = cfg.get("pso", {})
    print("min_velocity:", pso.get("min_velocity"))
    print("max_velocity:", pso.get("max_velocity"))
    
    mv = CONFIG["pso"]["min_velocity"]
    print(f"mv = CONFIG['pso']['min_velocity'] → {mv}")
    e = CONFIG["pso"]["boundary"]["clamp"]["epsilon"]
    print(f"e = CONFIGp['pso']['boundary']['clamp']['epsilon'] → {e}")
    ws = CONFIG["ann"]["weight_init_range"]
    print(f"ws = CONFIGp['ann']['weight_init_range'] → {ws}")
    print(f"ws[0] → {ws[0]}")
    print(f"ws[1] → {ws[1]}")
    