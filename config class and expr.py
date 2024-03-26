class Config:
    x = 1
    y = 1
    
    def __repr__(self):
        params = [f"{attr}: {getattr(self, attr)}" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return '\n'.join(params)
