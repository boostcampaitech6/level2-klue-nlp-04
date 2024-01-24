def modify_path_to_upper_directory(config: dict) -> dict:
    for key in config["path"].keys():
        if isinstance(config["path"][key], str):
            config["path"][key] = "." + config["path"][key]
    return config
