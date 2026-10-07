import torch
import typing as t

from nmln import ontology, utils


class Potential(torch.nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.cardinality = 1
        self.beta = torch.nn.Parameter(torch.Tensor([beta]), requires_grad=True)

    def forward(self, y):
        raise NotImplementedError

    def _reduce(self, y):
        return self.beta * self(y)


class GlobalPotential(torch.nn.Module):
    def __init__(self, potentials: t.List[Potential]):
        super().__init__()
        self.potentials = torch.nn.ModuleList(potentials)

    def forward(self, y):
        res = 0.0

        for potential in self.potentials:
            out = potential._reduce(y)
            # print("out", out.shape)
            res += out

        # res = res / len(self.potentials)

        return res


class CountableGroundingPotential(Potential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y):
        y = self.ground(y)
        y = self.call_on_groundings(y)
        y = self.reduce_groundings(y)

        return y

    def ground(self, y):
        raise NotImplementedError

    def call_on_groundings(self, y):
        raise NotImplementedError

    def reduce_groundings(self, y):
        raise NotImplementedError


class NeuralMLPPotential(CountableGroundingPotential):
    """Width-k neural local score: an MLP applied to every k-fragment of the
    world, summed over fragments."""

    @property
    def name(self):
        return f"k{self.k}"

    def __init__(
        self,
        k: int,
        potential_output_dim: int,
        ontology: ontology.Ontology,
        constants: t.List[str],
        sample_data,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.k = k
        self.potential_output_dim = potential_output_dim
        self.constants = constants

        self.groundings_hb_indices = torch.from_numpy(
            ontology.all_fragments_idx(k=k)
        )
        self.input_dim = self.ground_shape(sample_data)[-1]

        self.model = torch.nn.Sequential(
            *(
                [
                    torch.nn.Linear(self.input_dim, 150),
                    torch.nn.ReLU(),
                    torch.nn.Linear(150, 75),
                    torch.nn.ReLU(),
                    torch.nn.Linear(
                        75,
                        self.potential_output_dim,
                        bias=self.potential_output_dim > 1,
                    ),
                ]
                + ([torch.nn.ReLU()] if self.potential_output_dim > 1 else [])
            )
        )

    def ground(self, y):
        self.groundings_hb_indices = self.groundings_hb_indices.to(y.device)

        idx = self.groundings_hb_indices.reshape(
            [-1, self.groundings_hb_indices.shape[-1]]
        )

        return utils.tf_gather(y, idx, axis=-1)

    def ground_shape(self, y):
        return self.ground(y).shape

    def call_on_groundings(self, y):
        a = self.model(y)
        return a

    def reduce_groundings(self, y):
        # Aggregate fragments by summation.
        return y.squeeze(-1).sum(-1)
