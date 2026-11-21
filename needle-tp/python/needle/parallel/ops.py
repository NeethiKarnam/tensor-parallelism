import needle as ndl
from needle.autograd import Tensor, Value, TensorOp
import needle.backend_ndarray as nd
import needle.distributed as dist

class ParallelOp(TensorOp):
    pass

# NOTE: Below is a reference mockup of f and g operations. 
# Since distributed backend details like `all_reduce` depend on your needle distributed backend,
# adjust the communication calls (e.g., using ndl.cpu_numpy() or your custom MPI/NCCL wrappers) accordingly.

class F_Op(ParallelOp):
    """
    f operation:
    Forward: Identity.
    Backward: All-reduce gradient.
    """
    def compute(self, x):
        return x

    def gradient(self, out_grad, node):
        # Assuming your needle distributed module has all_reduce or similar
        import needle.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            return dist.all_reduce(out_grad)
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