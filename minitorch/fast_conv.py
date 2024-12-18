from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile a function using Numba."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.1.

    # Idea: batch represents the number of images in tne batch
    # out_channel represents the number of filters / potential features identifier
    # out_width represent the number of pixels in the output image (because we can do pooling on the input image)
    # in_channels represent for the same position of the image we can have different channels (such as RGB)
    # width represent the number of pixels in a image
    # weight_width represent the size of the filter

    # pseudo code:
    # for each batch:
    #   for each out_channel:
    #       for each out_width:
    #           for each in_channels:
    #               for each weight_width:
    #                   if not reverse:
    #                       anchor weight at left
    #                       for each width:
    #                           out[b, co, i] += input[b, ci, j] * weight[co, ci, wi]
    #                           wi += 1
    #                   else:
    #                       anchor weight at right
    #                       for each width:
    #                           out[b, co, i] += input[b, ci, j] * weight[co, ci, kw - 1 - (i - j)]
    #                           wi += 1

    # Step 1: for each batch
    for b in prange(batch_):
        # Step 2: Iterate over output channels
        for co in prange(out_channels):
            # Step 3: Iterate over output width
            for i in prange(out_width):
                # Step 4: Iterate over input channels

                out_pos = b * out_strides[0] + co * out_strides[1] + i * out_strides[2]

                acc = 0.0

                for ci in prange(in_channels):
                    # count the base index of input and weight and initialize the weight index
                    in_base = b * s1[0] + ci * s1[1]
                    weight_base = co * s2[0] + ci * s2[1]
                    wi = 0

                    if not reverse:
                        for j in prange(min(i, width - 1), min(i + kw, width)):
                            wi = j - i
                            acc += (
                                input[in_base + j * s1[2]]
                                * weight[weight_base + wi * s2[2]]
                            )
                            wi += 1
                    else:
                        for j in prange(max(i - kw + 1, 0), max(i + 1, width)):
                            wi = kw - 1 - (i - j)
                            if 0 <= wi < kw:
                                acc += (
                                    input[in_base + j * s1[2]]
                                    * weight[weight_base + wi * s2[2]]
                                )
                            wi += 1

                    # write the accumulated value to the output
                out[out_pos] = acc

    # Task 4.1 finished


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the convolution operation"""
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.
    for b in prange(batch):
        for co in prange(out_channels):
            for ho in prange(out_height):
                for wo in prange(out_width):
                    o = (
                        b * out_strides[0]
                        + co * out_strides[1]
                        + ho * out_strides[2]
                        + wo * out_strides[3]
                    )
                    for ci in prange(in_channels):
                        hw, ww = 0, 0
                        if not reverse:
                            h_start = min(ho, height - 1)
                            h_end = min(ho + kh, height)
                            w_start = min(wo, width - 1)
                            w_end = min(wo + kw, width)

                            for hi in prange(h_start, h_end):
                                for wi in prange(w_start, w_end):
                                    out[o] += (
                                        input[b * s10 + ci * s11 + hi * s12 + wi * s13]
                                        * weight[
                                            co * s20 + ci * s21 + hw * s22 + ww * s23
                                        ]
                                    )
                                    ww += 1
                                ww = 0
                                hw += 1
                        else:
                            h_start = max(ho - kh + 1, 0)
                            h_end = min(ho + 1, height)
                            w_start = max(wo - kw + 1, 0)
                            w_end = min(wo + 1, width)

                            for hi in prange(h_start, h_end):
                                for wi in prange(w_start, w_end):
                                    out[o] += (
                                        input[b * s10 + ci * s11 + hi * s12 + wi * s13]
                                        * weight[
                                            co * s20 + ci * s21 + hw * s22 + ww * s23
                                        ]
                                    )
                                    ww += 1
                                ww = 0
                                hw += 1


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the convolution operation"""
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
