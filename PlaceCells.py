import torch


class PlaceCells:
    def __init__(
        self,
        environment,
        npcs=512,
        pc_width=0.12,
        DoG=False,
        surround_scale=2,
        p=2.0,
        seed=0,
        dtype=torch.float32,
        **kwargs
    ):
        """
        Initialize the PlaceCells class.

        Parameters:
            environment (Environment object): The environment in which the agent is moving.
            npcs (int, optional): The number of place cells. Defaults to 512.
            pc_width (float, optional): The tuning curve width of the place cells. Defaults to 0.12.
            DoG (bool, optional): If True, apply Difference of Gaussians (DoG) to the place cell activities. Defaults to False.
            surround_scale (int, optional): The scale of the surround in the DoG. Defaults to 2.
            p (float, optional): The power for the distance calculation. Defaults to 2.0.
            seed (int, optional): The seed for random number generation. Defaults to 0.
            dtype (torch.dtype, optional): The data type for the place cell centers. Defaults to torch.float32.
        """
        self.environment = environment
        self.pcs = environment.sample_uniform(
            npcs, seed=seed
        )  # Uniformly sample place cell centers
        self.pcs = torch.tensor(self.pcs, dtype=dtype)  # Convert to tensor
        self.npcs = npcs
        self.pc_width = pc_width
        self.DoG = DoG
        self.surround_scale = surround_scale
        self.p = p
        self.seed = seed

    def softmax_response(self, pos):
        """
        Compute the place cell response as modelled by Sorscher.

        Parameters:
            pos (torch.Tensor): A tensor of shape (nsamples, 2) representing the cartesian positions to encode into place cell activities.
        Returns:
            torch.Tensor: A tensor of shape (nsamples, self.npcs) representing the place cell activities at the given spatial positions.
        """
        # Compute the distance to each place cell center
        dists = torch.sum(
            (pos[:, None] - self.pcs[None]) ** self.p, axis=-1
        )  # (nsamples, self.npcs)
        # Compute the softmax of the negative distances
        activities = torch.nn.functional.softmax(
            -dists / (2 * self.pc_width**2), dim=-1
        )
        if self.DoG:
            # Apply Difference of Gaussians (DoG) to the activities
            activities -= torch.nn.functional.softmax(
                -dists / (2 * self.surround_scale * self.pc_width**2), dim=-1
            )
            # Normalize the activities to make them a probability distribution again
            activities -= torch.min(activities, dim=-1, keepdim=True).values
            activities /= torch.sum(activities, dim=-1, keepdim=True)
        return activities

    def to_euclid(self, activities, k=3):
        """
        Decode place-cell activities to Euclidean coordinates following Sorscher's method.
        Note: This is an approximation to the actual Euclidean location, by considering
        the top k place-cell activities as if the agent is located at the average k place-cell center location.

        Parameters:
            activities (torch.Tensor): A (nsamples,self.npcs) tensor of place cell activities.
            k (int, optional): The number of top place-cell activities to consider. Defaults to 3.
        Returns:
            torch.Tensor: The decoded Euclidean coordinates.
        """
        # Get the indices of the top k activities
        _, idxs = torch.topk(activities, k, dim=-1)
        # Return the mean of the corresponding place cell centers
        return torch.mean(self.pcs[idxs], axis=-2)
