from typing import Tuple
from einops import rearrange
import numpy as np
import torch


class RandomPointSampler3D:
    def __init__(
        self,
        coordinates: torch.Tensor,
        data: torch.Tensor,
        weight_map: torch.Tensor,
        n_points_per_sampling: int,
    ) -> None:
        self.n_points_per_sampling = n_points_per_sampling
        self.flattened_coordinates = rearrange(coordinates, "d h w c-> (d h w) c")
        self.flattened_data = rearrange(data, "d h w c-> (d h w) c")
        self.flattened_weight_map = rearrange(weight_map, "d h w c-> (d h w) c")
        self.n_total_points = self.flattened_data.shape[0]

    def next(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map


class RandomPointSampler4D:
    def __init__(
        self,
        coordinates: torch.Tensor,
        data: torch.Tensor,
        weight_map: torch.Tensor,
        n_points_per_sampling: int,
    ) -> None:
        self.n_points_per_sampling = n_points_per_sampling
        self.flattened_coordinates = rearrange(coordinates, "t d h w c-> (t d h w) c")
        self.flattened_data = rearrange(data, "t d h w c-> (t d h w) c")
        self.flattened_weight_map = rearrange(weight_map, "t d h w c-> (t d h w) c")
        self.n_total_points = self.flattened_data.shape[0]

    def next(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_idxs = torch.randint(
            0, self.n_total_points, (self.n_points_per_sampling,), device="cuda"
        )
        sampled_coords = self.flattened_coordinates[sampled_idxs, :]
        sampled_data = self.flattened_data[sampled_idxs, :]
        sampled_weight_map = self.flattened_weight_map[sampled_idxs, :]
        return sampled_coords, sampled_data, sampled_weight_map
