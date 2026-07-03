from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class Polygon:
    # use shapely.geometry lib
    def __init__():
        pass


class SafetyScenario(str, Enum):
    LANE_KEEP = "LANE_KEEP"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    MERGE = "MERGE"
    FREE_SPACE = "FREE_SPACE"
    UNKNOWN = "UNKNOWN"


class SafetyRewardCalculator:
    def __init__(self, coner_points: List):
        self.coner_points = coner_points # [5, 2]

    def _extract_sample_data(
        self, data_dict: Dict, pred_dict: Dict, batch_index: int
    ) -> Tuple[Dict, Dict]:
        data_dict = {key: value[batch_index] for key, value in data_dict.items()}
        pred_dict = {key: value[batch_index] for key, value in pred_dict.items()}
        return data_dict, pred_dict

    def _get_scenario(ego_trajectories: torch.Tensor) -> List[SafetyScenario]:
        """Get scenario based on ego future and lane groups for each mode."""

    def _get_roi_agent_ids(
        self,
        scenario: List[SafetyScenario],
    ) -> torch.Tensor:
        """Get the region of interest based on scenario using blacklist approach.

        Returns:
            torch.Tensor: The ROI agent ids for each mode, the shape is [M, num_roi_agents]
        """

    def _filter_agents_by_roi(
        agents_trajectory: torch.Tensor, roi_agent_ids: torch.Tensor
    ):
        pass

    def _get_ego_polygons(self, ego_trajectories: torch.Tensor) -> List[List[Polygon]]:
        """
        Args:
            ego_trajectories: shape [M, T, 13], formed as:
            [x, y, l, w, yaw, vx, vy, ax, ay, d2front, d2rear, d2left, d2right].
        """
        ego_box = torch.zeros(
            (ego_trajectories.shape[0], ego_trajectories.shape[1], 5, 2)
        )
        yaw = ego_trajectories[..., 4]
        x = ego_trajectories[..., 0]
        y = ego_trajectories[..., 1]
        d2front = ego_trajectories[..., 9]
        d2rear = ego_trajectories[..., 10]
        d2left = ego_trajectories[..., 11]
        d2right = ego_trajectories[..., 12]

        x1 = x + d2front * torch.cos(yaw)
        y1 = y + d2front * torch.sin(yaw)

        x2 = x + d2rear * torch.cos(yaw + torch.pi)
        y2 = y + d2rear * torch.sin(yaw + torch.pi)

        ego_box[..., 0, 0] = x1 + d2right * torch.cos(yaw - torch.pi / 2)
        ego_box[..., 0, 1] = y1 + d2right * torch.sin(yaw - torch.pi / 2)

        ego_box[..., 1, 0] = x1 + d2left * torch.cos(yaw + torch.pi / 2)
        ego_box[..., 1, 1] = y1 + d2left * torch.sin(yaw + torch.pi / 2)

        ego_box[..., 2, 0] = x2 + d2left * torch.cos(yaw + torch.pi / 2)
        ego_box[..., 2, 1] = y2 + d2left * torch.sin(yaw + torch.pi / 2)

        ego_box[..., 3, 0] = x2 + d2right * torch.cos(yaw + 3 * torch.pi / 2)
        ego_box[..., 3, 1] = y2 + d2right * torch.sin(yaw + 3 * torch.pi / 2)

        # Close the rectangle by appending the first point at the end
        ego_box[..., 4, 0] = ego_box[..., 0, 0]
        ego_box[..., 4, 1] = ego_box[..., 0, 1]

        ego_polygon_list = []
        for egos in ego_box:
            ego_polygons = [Polygon(ego.tolist()) for ego in egos]
            ego_polygon_list.append(ego_polygons)

        return ego_polygon_list

    def _get_agents_polygons(
        self, agents_trajectory: torch.Tensor
    ) -> List[List[List[Polygon]]]:
        """
        Args:
            agents_trajectory: shape [M, N, T, 12], formed as
            [x, y, l, w, yaw, vx, vy, ax, ay, objtype, signal_light, brake_light].
        """

    def _get_ego_s(ego_trajectories: torch.Tensor):
        pass

    def _get_agents_frenet_sl(
        ego_trajectories: torch.Tensor, agents_trajectory: torch.Tensor
    ):
        pass

    def _get_dynamic_object_cost(
        ego_trajectories: torch.Tensor,  # [M, T, 13]
        ego_polygons: List[List[Polygon]],  # [M, T]
        agents_trajectory: torch.Tensor,  # [M, N, T, 12]
        agents_polygons: Optional[List[List[List[Polygon]]]], # [M, N, T]
        ego_s: torch.Tensor, # [M, T]
        agents_sl: Optional[torch.Tensor], # [M, N, T, 2]
    ) -> torch.Tensor:
        pass

    def get_reward(self, data_dict: Dict, pred_dict: Dict) -> torch.Tensor:
        """Computes safety reward based on the given input data and sampled trajectory.

        Returns:
            torch.Tensor: safety reward value for each trajectory point, the shape is [B, N, T]
        """

        batch_size, num_modalities, time_steps = pred_dict["predicted_trajectory"]
        costs = torch.zeros(
            (batch_size, num_modalities, time_steps), dtype=torch.float32
        )

        batch_scenarios = []
        batch_roi_agents = []

        for i in range(batch_size):
            data, pred = self._extract_sample_data(data_dict, pred_dict, i)

            ego_trajectories = pred["predicted_trajectory"]
            agents_trajectory = data["agents_trajectory"]

            scenario = self._get_scenario(ego_trajectories)
            batch_scenarios.append(scenario)

            roi_agent_ids = self._get_roi_agent_ids(scenario, ego_trajectories)
            batch_roi_agents.append(roi_agent_ids)

            agents_trajectory = self._filter_agents_by_roi(
                agents_trajectory, roi_agent_ids
            )

            ego_polygons = self._get_ego_polygons(ego_trajectories)
            agents_polygons = self._get_agents_polygons(agents_trajectory)
            ego_s = self._get_ego_s(ego_trajectories)
            agents_sl = self._get_agents_frenet_sl(ego_trajectories, agents_trajectory)

            dynamic_object_cost = self._get_dynamic_object_cost(
                ego_trajectories,
                ego_polygons,
                agents_trajectory,
                agents_polygons,
                ego_s,
                agents_sl,
            )
            costs[i] = dynamic_object_cost

        reward = -costs
        return reward
