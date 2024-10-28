from fugep.setup import load_path, parse_configs_and_run


configs = load_path("../configs/config.yml")
parse_configs_and_run(configs, lr=0.2)