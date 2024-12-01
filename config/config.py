import json
import inspect


class Config:
    def to_dict(self):
        serializable_dict = {}
        for key, value in self.__dict__.items():
            try:
                json.dumps(value)
                serializable_dict[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable attributes
                pass
        return serializable_dict

    @classmethod
    def from_dict(cls, config_dict):
        init_params = inspect.signature(cls.__init__).parameters
        valid_params = {k: v for k, v in config_dict.items() if k in init_params}
        return cls(**valid_params)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
