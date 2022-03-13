import copy
import torch
from torch.autograd import Function
from torch.nn import Module


class WeightGradientsFunc(Function):
    """
    Gradient weighting and reversal function used by the WeightGradients module.
    """

    @staticmethod
    def forward(ctx, input_, grad_weight):
        ctx.save_for_backward(input_)
        ctx.grad_weight = grad_weight
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_weight = ctx.grad_weight
        # grad_weight = grad_output.new_tensor(grad_weight)
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_weight * grad_output

        return grad_input, None


class WeightGradients(Module):
    def __init__(self, grad_weight=-1, *args, **kwargs):
        """
        The layer can be used to weight and reverse gradients in the backward pass.
        For gradient reversal, grad_weight should be negative (-1 for pure reversal).
        """

        super(WeightGradients, self).__init__(*args, **kwargs)
        self.grad_weight = grad_weight

    def forward(self, input_):
        return WeightGradientsFunc.apply(input_, self.grad_weight)


def test_weight_gradients(grad_weight):
    """
    Test for WeightGradients module.
    """

    network = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Linear(3, 1))
    reverse_network = torch.nn.Sequential(copy.deepcopy(network), WeightGradients(grad_weight=grad_weight))

    inp = torch.randn(8, 5)
    outp = torch.randn(8)

    criterion = torch.nn.MSELoss()

    criterion(network(inp), outp).backward()
    criterion(reverse_network(inp), outp).backward()

    print(network.parameters())

    for p1, p2 in zip(network.parameters(), reverse_network.parameters()):
        print(p1.grad)
        print(p2.grad)

    assert all(
        (grad_weight * p1.grad == p2.grad).all()
        for p1, p2 in zip(network.parameters(), reverse_network.parameters())
    )
