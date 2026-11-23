"""Tests verifying all baseline and ablation schedulers."""

import numpy as np

from eatsched.agents import (
    D2SACSAAgent,
    D2SACScheduler,
    DDPGScheduler,
    DPPOScheduler,
    EATCAAgent,
    EATDAgent,
    EDFScheduler,
    FIFOScheduler,
    GreedyScheduler,
    PPOScheduler,
    PriorityFIFOScheduler,
    RandomScheduler,
    SACScheduler,
    TD3Scheduler,
)
from eatsched.env.sched_env import CloudEdgeSchedEnv


def test_all_schedulers_valid_action():
    """Verify that every baseline and ablation outputs valid actions within [-1, 1]^3."""
    env = CloudEdgeSchedEnv(cluster_name="edge32", mode="paper_mode", seed=42)
    obs, _ = env.reset(seed=42)

    schedulers = [
        RandomScheduler(seed=42),
        FIFOScheduler(),
        GreedyScheduler(),
        EDFScheduler(),
        PriorityFIFOScheduler(),
        SACScheduler(),
        PPOScheduler(),
        TD3Scheduler(),
        DDPGScheduler(),
        DPPOScheduler(),
        D2SACScheduler(),
        EATDAgent(),
        EATCAAgent(),
        D2SACSAAgent(),
    ]

    for sched in schedulers:
        if isinstance(sched, GreedyScheduler):
            action = sched.select_action(obs, env=env)
        else:
            action = sched.select_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
