import ConfigParser
import sys
import collections


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_config(config_section, config_path):
    """
    Read a config file into a dictionary, while converting variables to appropriate types
    """
    config_parser = ConfigParser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)

    config = collections.OrderedDict()

    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)

    return config

def print_config(config):
    for key, value in config.items():
        print(str(key) + ": " + str(value))
