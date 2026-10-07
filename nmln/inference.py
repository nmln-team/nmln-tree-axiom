import torch
import random
import typing as t

from nmln import (
    utils,
    constraints,
    potentials,
)


class Sampler:
    def sample(self):
        raise NotImplementedError


class GPUGibbsSampler(Sampler):
    def __init__(
        self,
        global_potential: potentials.GlobalPotential,
        constraints: t.List[constraints.NewOneOfNConstraint],
        num_variables: int,
        num_chains: int,
        num_examples: int,
        seed: int = 0,
        inter_sample_burn: int = 1,
        log_function: t.Optional[t.Callable[..., None]] = None,
    ):
        super().__init__()

        # Initialize instance variables with provided arguments
        self.global_potential = global_potential
        self.num_variables = num_variables  # Total number of variables in the model
        self.num_chains = num_chains  # Number of Markov chains to run in parallel
        self.num_examples = num_examples  # Number of examples to sample
        self.constraints = constraints  # Constraints to be applied during sampling
        # Own RNG stream. The global `random` module is also drawn from by
        # libraries in the training stack a variable number of times, which
        # would otherwise make the constraint order irreproducible.
        self._rng = random.Random(seed)
        self.inter_sample_burn = (
            inter_sample_burn  # Number of burn-in steps between samples
        )
        # Initialize the current state tensor with zeros
        # Dimensions: [num_examples, num_chains, num_variables]
        self.current_state = torch.zeros(
            [self.num_examples, self.num_chains, self.num_variables],
            dtype=torch.float32,
        )
        self.log_function = log_function

    def _inner_sample(self, y, m, alsoNone):
        """
        Helper function to prepare tensors for sampling based on constraints.

        Args:
            y (tensor): Current state tensor.
            m (tensor): Mask tensor from constraints.
            alsoNone (bool): Indicates if "None" option is allowed in constraints.

        Returns:
            tensor: Transformed tensor ready for sampling.
        """
        # Sum over the last but one dimension of the mask to identify masked positions
        m2 = m.sum(-2)
        # Expand dimensions of the mask to match the dimensions of the current state
        M = m.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Create a tensor 't' where positions with constraints are set to zero
        t = torch.where(m2 > 0, torch.zeros_like(y), y)
        # Clone 't' to 't1' for possible use when 'alsoNone' is True
        t1 = torch.clone(t)

        # Add a new dimension at position -2 to 't' for concatenation
        t = t.unsqueeze(-2)
        # Set positions where mask 'M' is active to ones in 't'
        t = torch.where(M > 0, torch.ones_like(t), t)

        if alsoNone:
            # If 'alsoNone' is True, include the option of 'None' in sampling
            # Expand 't1' to match the shape of 't'
            t1 = t1.unsqueeze(-2)
            t1 = t1.expand(
                t.shape[0], t.shape[1], t.shape[2], t.shape[3], t.shape[4], t.shape[5]
            )
            # Concatenate 't1' and 't' along the -2 dimension to include 'None' option
            t = torch.cat((t1, t), -2)

        return t  # Return the prepared tensor for sampling

    def _sample(self, current_state):
        """
        Perform a single blocked Gibbs sampling step for one global potential.

        Args:
            current_state (tensor): The current state of the chains.
            global_potential (callable): The global potential function to use.

        Returns:
            tensor: Updated current state after sampling.
        """
        # Randomly shuffle the constraints to avoid systematic biases
        for constraint in self._rng.sample(self.constraints, len(self.constraints)):
            if isinstance(constraint, constraints.TreeConstraint):
                current_state = constraint.gibbs_sampling_step(
                    current_state, self.global_potential
                )

            else:
                # Iterate over the binary masks of the constraint
                for mask in constraint.binary_masks:
                    # Prepare the sampling parameters using the helper function
                    params = self._inner_sample(current_state, mask, constraint.alsoNone)
                    params_shaped = params.reshape([-1, params.shape[-1]])
                    # Compute the potentials (energies) for the prepared parameters
                    potentials = self.global_potential(params_shaped)
                    # Reshape potentials back to match the shape of 'params' (excluding the last dimension)
                    potentials = potentials.reshape(params.shape[:-1])

                    # Reshape potentials to compute logits for the categorical distribution
                    logits = potentials.reshape([-1, potentials.shape[-1]])
                    # Apply log softmax to get normalized log-probabilities
                    logits = torch.log_softmax(logits, dim=-1)

                    # Sample from the categorical distribution based on the logits
                    # sample = torch.distributions.categorical.Categorical(
                    #     logits=logits
                    # ).sample()
                    # Should be faster than Categorical above.
                    gumbels = -torch.empty_like(logits).exponential_().log()
                    sample = (logits + gumbels).argmax(dim=-1)
                    # Reshape the sample to match the original shape before reshaping
                    sample = sample.reshape([*potentials.shape[:-1], 1])

                    # Update the current state based on the sampled indices
                    # 'tf_gather' gathers values from 'params' based on 'sample' indices
                    current_state = utils.tf_gather_o1(params, sample, batch_dims=4)
                    # Remove the singleton dimension added earlier
                    current_state = current_state.squeeze(-2)

        return current_state  # Return the updated current state after sampling

    def sample(self):
        """
        Generate samples with blocked Gibbs sampling.

        Returns:
            tensor: Samples of shape [num_examples, num_chains, num_variables].
        """
        device = next(self.global_potential.parameters()).device
        self.current_state = self.current_state.to(device)

        for _ in range(self.inter_sample_burn):
            self.current_state = self._sample(self.current_state)

        return self.current_state
