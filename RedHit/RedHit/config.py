import yaml


class RedHitConf:
    filepath = '/home/researchuser/LLMSec/LLMSecurity/RedHit/config.yml'

    @classmethod
    def read_config(cls):
        try:
            with open(cls.filepath, 'r') as file:
                config = yaml.safe_load(file) or {}
            return config
        except FileNotFoundError:
            print("Error: RedHit config file not found")
            raise

    @classmethod
    def get(cls, key):
        if not hasattr(cls, '_variables'):
            cls._variables = cls.read_config()

        return cls._variables.get(key)
