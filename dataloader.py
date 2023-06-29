import numpy as np
import torch
from typing import Tuple


class Dataset(torch.utils.data.Dataset):
    def __init__(self, agent, place_cells, num_samples: int, seq_len: int = 20) -> None:
        """
        Initialize the dataset.

        Parameters:
            agent: The agent object.
            place_cells: The place cells object.
            num_samples: The number of samples in the dataset.
            seq_len: The sequence length.
        """
        self.agent = agent
        self.place_cells = place_cells
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            The number of samples.
        """
        return self.num_samples

    def __getitem__(
        self, index: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Parameters:
            index: The index of the sample (unused).

        Returns:
            velocities (self.seq_len,2): cartesian velocities
            init_pc_positions (npcs,): initial place cell activities
            labels (self.seq_len, npcs): true place cell activities
            positions (self.seq_len, 2): true cartesian positions (not used for training)
        """
        self.agent.reset()
        for _ in range(self.seq_len):
            self.agent.step()
        velocities = torch.tensor(self.agent.velocities[1:], dtype=torch.float32)
        positions = torch.tensor(self.agent.positions, dtype=torch.float32)
        pc_positions = self.place_cells.softmax_response(positions)
        init_pc_positions, labels = pc_positions[0], pc_positions[1:]
        return velocities, init_pc_positions, labels, positions
