"""
TODO: write this
"""

import torch

from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_primitives import choose_qparams_affine, MappingType
from torchao.dtypes.utils import (
    LayoutType,
    PlainLayoutType,
)

from my_dtype_tensor_subclass import (
    MyDTypeLayout,
    MyDTypeTensor,
)

from torchao.utils import (
    _get_layout_tensor_constructor,
    _register_layout_cls,
)

aten = torch.ops.aten


##############################
# Tensor Subclass Definition #
##############################

class _ToMyTrainableDTypeTensor(torch.autograd.Function):
    """ 
    Differentiable constructor for `MyTrainableDTypeTensor`.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_float: torch.Tensor,
        layout_type: LayoutType,
    ) -> "MyTrainableDTypeTensor":
        mapping_type = MappingType.SYMMETRIC
        block_size = input_float.shape
        dtype = torch.int16
        scale, _ = choose_qparams_affine(input_float, mapping_type, block_size, dtype)
        int_data = (input_float / scale).to(torch.int8)
        layout_tensor_ctr = get_layout_tensor_constructor(type(layout_type))
        layout_tensor = layout_tensor_ctr(int_data, scale, layout_type)
        return MyTrainableDTypeTensor(layout_tensor, input_float.shape)

    @staticmethod
    def backward(ctx, gy):
        return gy, None

class MyTrainableDTypeTensor(MyDTypeTensor):
    """
    TODO: write this
    """

    """classmethod that converts from a floating point Tensor (fp32/fp16/bf16) to the current dtype
    """

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        layout_type: LayoutType = PlainLayoutType(),
    ):
        return _ToMyTrainableDTypeTensor.apply(input_float, layout_type)


######################################################
# LayoutType and Layout Tensor Subclass Registration #
######################################################

def register_layout_cls(layout_type_class: type(LayoutType)):
    return _register_layout_cls(MyTrainableDTypeTensor, layout_type_class)

def get_layout_tensor_constructor(layout_type_class: type(LayoutType)):
    return _get_layout_tensor_constructor(MyTrainableDTypeTensor, layout_type_class)


@register_layout_cls(PlainLayoutType)
class PlainMyDTypeLayout(MyDTypeLayout):
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        self.int_data = int_data
        self.scale = scale
        self.layout_type = layout_type

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [self.layout_type]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale = tensor_data_dict["int_data"], tensor_data_dict["scale"]
        layout_type, = tensor_attributes
        return cls(int_data, scale, layout_type)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        layout_type: LayoutType,
    ):
        """Construct a layout tensor from plain tensors and a layout_type, which main contain
        extra metadata for packing etc.
        """
        assert isinstance(layout_type, PlainLayoutType)
        return cls(int_data, scale, layout_type)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            self.layout_type,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"MyTrainableDTypeLayout dispatch: attempting to run {func}, this is not supported"
        )

#####################################################
# torch functional and aten operator implementation #
#####################################################

implements = MyTrainableDTypeTensor.implements

class _QuantizedLinearOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> "MyTrainableDTypeTensor":
        ctx.save_for_backward(input_tensor, weight_tensor)
        if isinstance(input_tensor, MyTrainableDTypeTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, MyTrainableDTypeTensor):
            weight_tensor = weight_tensor.dequantize()
        return torch.nn.functional.linear(input_tensor, weight_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight_tensor = ctx.saved_tensors
        print("hi I'm backward", type(input_tensor), type(weight_tensor))
        raise NotImplementedError("BACKWARD not implemented")
        #grad_input = (grad_output * weight_tensor.scale) @ weight.int_data.to(grad_output.dtype)
        #grad_weight = grad_output.view(-1, weight.shape[0]).T @ input.view(-1, weight.shape[1])
        #return grad_input, grad_weight


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    return _QuantizedLinearOp.apply(input_tensor, weight_tensor)


class M(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1024, 1024, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

#####################
# Factory functions #
#####################
to_my_trainable_dtype = MyTrainableDTypeTensor.from_float


########
# Test #
########
from torchao.utils import benchmark_model

m = M().cuda()
example_inputs = (100 * torch.randn(1024, 1024).cuda(),)
NUM_WARMUPS = 10
NUM_RUNS = 100

for _ in range(NUM_WARMUPS):
    m(*example_inputs)

#compiled = torch.compile(m, mode="max-autotune")
#for _ in range(NUM_WARMUPS):
#    compiled(*example_inputs)

# convert weights to quantized weights
m.linear.weight = torch.nn.Parameter(
    to_my_trainable_dtype(m.linear.weight), requires_grad=True,
)

for _ in range(NUM_WARMUPS):
    m(*example_inputs)

optimizer = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(100):
    target = torch.randn(1024).long().cuda()
    output = m(*example_inputs)
    print("shapey shape", output.shape, output.dtype, output.device)
    loss = loss_fn(output, target)
    loss.backward()
    if i < 5 or i > 95: 
        print(" -- step", i)
    #    print(
    #        " -- step", i,
    #        "weight type =", type(weight).__name__,
    #        "requires_grad =", weight.requires_grad,
    #        "grad_value =", weight.grad.flatten()[:3] if weight.grad is not None else "<no_grad>",
    #        "value =", orig.flatten()[:3],
    #    )   
    optimizer.step()
    optimizer.zero_grad()

#print("after quantization:", benchmark_model(m, NUM_RUNS, example_inputs[0]))
#
#m = torch.compile(m, mode="max-autotune")
#
#for _ in range(NUM_WARMUPS):
#    m(*example_inputs)
#
## NOTE: currently there is no speedup because we just dequantize the weight in the _quantized_linear op
## we plan to add custom op example in the future and that will help us to get speedup
#print("after quantization and compile:", benchmark_model(m, NUM_RUNS, example_inputs[0]))
