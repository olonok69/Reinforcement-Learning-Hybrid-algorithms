"""benchmarks — public API for the RL Continuous Control benchmark suite.

Re-exports the configuration dataclasses, training entry points, and shared
utilities from each algorithm module so callers can import from a single location::

    from benchmarks import DDPGConfig, run_ddpg, run_timed, set_global_seed

Modules
-------
common          Shared utilities: BenchmarkResult, metrics helpers, seed control.
replay_buffer   Off-policy experience store shared by DDPG, TD3, and SAC.
ddpg            Deep Deterministic Policy Gradient implementation.
td3             Twin Delayed DDPG implementation.
sac             Soft Actor-Critic implementation.
"""

from .common import BenchmarkResult
from .common import run_timed
from .common import save_results_csv
from .common import save_results_json
from .common import set_global_seed
from .ddpg import DDPGConfig, run_ddpg
from .td3 import TD3Config, run_td3
from .sac import SACConfig, run_sac
