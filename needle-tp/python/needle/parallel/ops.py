import needle as ndl
from needle.autograd import Tensor, Value, TensorOp
import needle.backend_ndarray as nd
import needle.parallel.distributed as dist

class ParallelOp(TensorOp):
    pass

# dist.all_reduce takes and returns raw NDArrays (see ROADMAP.md, Interfaces §1).
# compute() already receives NDArrays; gradient() works on Tensors, so it unwraps
# with realize_cached_data() and rewraps the result with Tensor.make_const().

class F_Op(ParallelOp):
    """
    f operation:
    Forward: Identity.
    Backward: All-reduce gradient.
    """
    def compute(self, x):
        return x

    def gradient(self, out_grad, node):
        if dist.is_initialized() and dist.get_world_size() > 1:
            return Tensor.make_const(dist.all_reduce(out_grad.realize_cached_data()))
        return out_grad

class G_Op(ParallelOp):
    """
    g operation:
    Forward: All-reduce input.
    Backward: Identity.
    """
    def compute(self, x):
        if dist.is_initialized() and dist.get_world_size() > 1:
            return dist.all_reduce(x)
        return x

    def gradient(self, out_grad, node):
        return out_grad

def f(x):
    return F_Op()(x)

def g(x):
    return G_Op()(x)