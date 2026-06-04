import inspect

from .strategies import DistillationStrategy, RADStrategy, VADStrategy, IRBCStrategy, BSDStrategy
from .trainer import DistillationTrainer

STRATEGIES = {
    "rad": RADStrategy,
    "vad": VADStrategy,
    "irbc": IRBCStrategy,
    "bsd": BSDStrategy,
}


def get_strategy(name, **kwargs):
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}"
        )
    cls = STRATEGIES[name]
    params = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return cls(**filtered)
