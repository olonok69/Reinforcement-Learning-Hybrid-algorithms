"""replay_buffer — off-policy experience store shared by DDPG, TD3, and SAC.

Design notes
------------
* Pre-allocated fixed-size NumPy arrays avoid dynamic memory allocation during training.
* A circular write pointer overwrites the oldest transitions once capacity is reached.
* Rewards and ``not_done`` flags are stored as (N,1) columns so they broadcast cleanly
  against Q-value tensors of shape (batch, 1) inside the Bellman targets.
* The buffer stores ``not_done = 1 - done`` (the complement) rather than ``done`` itself.
  This lets every algorithm write the Bellman target as::

      target_q = reward + not_done * gamma * bootstrap_value

  without any conditional logic inside the vectorised update.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular experience replay buffer for off-policy RL algorithms.

    Stores transitions ``(state, action, next_state, reward, done)`` in
    contiguous NumPy arrays and returns uniformly sampled mini-batches as
    PyTorch tensors ready for gradient computation.

    Parameters
    ----------
    state_dim:
        Dimensionality of the observation/state vector.
    action_dim:
        Dimensionality of the action vector.
    max_size:
        Maximum number of transitions to hold.  Older entries are overwritten
        once the buffer is full.
    device:
        PyTorch device for the tensors returned by :meth:`sample`.  Defaults to
        CUDA if available, otherwise CPU.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6), device: torch.device | None = None):
        self.max_size = max_size
        self.ptr = 0    # write cursor; wraps around at max_size
        self.size = 0   # number of valid transitions; saturates at max_size

        # Pre-allocate all arrays once to avoid per-step allocation overhead.
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        # Stored as (1 - done) so the Bellman target can be computed without branching:
        #   target = r + not_done * gamma * bootstrap
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        """Write one transition into the buffer.

        If the buffer is full the entry at ``ptr`` (the oldest) is silently
        overwritten and the circular pointer advances.

        Parameters
        ----------
        state:
            Observation before the action was taken.
        action:
            Action executed by the agent.
        next_state:
            Observation received after the action.
        reward:
            Scalar reward signal returned by the environment.
        done:
            ``True`` if the episode terminated or was truncated after this step.
            Stored as ``not_done = 1 - done`` to simplify Bellman bootstrap.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)  # complement: 0.0 when terminal

        self.ptr = (self.ptr + 1) % self.max_size   # advance and wrap
        self.size = min(self.size + 1, self.max_size)  # cap at capacity

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a uniformly sampled mini-batch as PyTorch tensors.

        Sampling is uniform over all valid transitions, which breaks the
        temporal correlation present in on-policy rollout data and satisfies
        the i.i.d. assumption required for SGD-based training.

        Parameters
        ----------
        batch_size:
            Number of transitions to include in the mini-batch.

        Returns
        -------
        tuple of five tensors, each of shape ``(batch_size, dim)``:
            * ``states``      — shape ``(B, state_dim)``
            * ``actions``     — shape ``(B, action_dim)``
            * ``next_states`` — shape ``(B, state_dim)``
            * ``rewards``     — shape ``(B, 1)``
            * ``not_dones``   — shape ``(B, 1)``,  ``1.0`` if non-terminal, ``0.0`` if terminal
        """
        idx = np.random.randint(0, self.size, size=batch_size)  # uniform random indices
        return (
            torch.tensor(self.state[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.action[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.reward[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.not_done[idx], dtype=torch.float32, device=self.device),
        )
