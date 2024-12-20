from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.

    new_h, new_w = height // kh, width // kw

    outputTensor = input.contiguous()
    # reshape to batch x channel x h x kh x w x kw
    outputTensor = outputTensor.view(batch, channel, new_h, kh, new_w, kw)
    # reshape to batch x channel x h x kh x width
    outputTensor = outputTensor.permute(0, 1, 2, 4, 3, 5)
    # permute to batch x channel x h x w x kh
    outputTensor = outputTensor.contiguous()
    # make contiguous
    outputTensor = outputTensor.view(batch, channel, new_h, new_w, kh * kw)
    # reshape to batch x channel x h x w x kh * kw
    return outputTensor, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input, new_h, new_w = tile(input, kernel)
    return (
        input.mean(dim=4)
        .contiguous()
        .view(input.shape[0], input.shape[1], new_h, new_w)
    )


# TODO: Implement for Task 4.3.


class Max(Function):
    @staticmethod
    def forward(ctx: Context, inputTensor: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation"""
        ctx.save_for_backward(inputTensor, dim)
        return inputTensor.f.max_reduce(inputTensor, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operation"""
        input, dim = ctx.saved_tensors
        return grad_output * argmax(input, dim), 0.0


def argmax(inputTensor: Tensor, dim: int) -> Tensor:
    """Use argmax to implement one hot encoding"""
    return inputTensor == max(inputTensor, dim)


def max(inputTensor: Tensor, dim: int) -> Tensor:
    """Use max to implement argmax"""
    return Max.apply(inputTensor, inputTensor._ensure_tensor(dim))


def softmax(inputTensor: Tensor, dim: int) -> Tensor:
    """Use softmax to implement softmax"""
    return inputTensor.exp() / inputTensor.exp().sum(dim)


def logsoftmax(inputTensor: Tensor, dim: int) -> Tensor:
    """Use logsoftmax to implement softmax"""
    return softmax(inputTensor, dim).log()


def maxpool2d(inputTensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tile the input tensor and apply max pooling"""
    tiledTensor, new_h, new_w = tile(inputTensor, kernel)
    return (
        max(tiledTensor, dim=4)
        .contiguous()
        .view(tiledTensor.shape[0], tiledTensor.shape[1], new_h, new_w)
    )


def dropout(inputTensor: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off"""
    if ignore:
        return inputTensor
    else:
        rand_values = rand(inputTensor.shape, backend=inputTensor.backend)
        mask = rand_values > rate
        return inputTensor * mask / (1 - rate) if rate != 1.0 else inputTensor * mask
