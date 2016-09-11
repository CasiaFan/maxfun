from logging.config import dictConfig
import logging

logging_config = {
    "version": 1,
    "diable_existing_loggers": False,
    "formatters": {
        "f":{
            "format": "[%(asctime)s] %(levelname)s - %(name)s:%(lineno)d:%(message)s",
        },
    },
    "handlers": {
        "h": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "f",
        },
    },
    "loggers":{
        "l": {
            "level": "INFO",
            "handler": "h",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": "h",
    },
}

def get_logger(module_name=__name__, config=logging_config):
    # module_name: get current module name
    # config: the log configuration dictionary
    dictConfig(logging_config)
    logger = logging.getLogger(module_name)
    return logger
