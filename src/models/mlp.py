import torch
import numpy as np

class Mlp(torch.nn.Module):
    def __init__(
            self,
            input_size,
            layers_data,
            output_size,
            activation_fn=torch.nn.ReLU,
            scale_bias=1.0,
            scale_weights=1.0,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.input_size = input_size
        for size in layers_data:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(activation_fn())
            input_size = size
        self.output_layer = torch.nn.Linear(input_size, output_size)
        self.output_layer.weight.data.mul_(scale_weights)
        self.output_layer.bias.data.mul_(scale_bias)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        output_data = self.output_layer(input_data)
        return output_data
    
class PARAFAC(torch.nn.Module):
    """This class is only introduced to experiment 9, as it was requested  as a follow up"""
    def __init__(self, dims: np.ndarray, k: int, scale: float = 1.0, nA: int = 1, bias=0.0) -> None:
        super().__init__()

        self.nA = nA
        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) - bias) 
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices: np.ndarray) -> torch.Tensor:
        prod = torch.ones(self.k, dtype=torch.double)
        for i in range(len(indices)):
            idx = indices[i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if len(indices) < len(self.factors):
            res = []
            for cols in zip(
                *[self.factors[-(a + 1)].t() for a in reversed(range(self.nA))]
            ):
                kr = cols[0]
                for j in range(1, self.nA):
                    kr = torch.kron(kr, cols[j])
                res.append(kr)
            factors_action = torch.stack(res, dim=1)
            return torch.matmul(prod, factors_action.T)
        return torch.sum(prod, dim=-1)
