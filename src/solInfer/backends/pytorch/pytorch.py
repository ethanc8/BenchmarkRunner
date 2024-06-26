# SPDX-License-Identifier: AGPL-3.0-or-later
from solInfer import backends
import torch

class Backend_class(backends.Backend):
    def __init__(self):
        pass

class Tensor(backends.Tensor):
    def __init__(self, data: torch.Tensor):
        self.data = data
    
    ## Indexing ##

    def __getitem__(self, key):
        res = self.data[key]
        if isinstance(res, torch.Tensor):
            return self.scalarize_or_tensorize(res)
        else:
            return res

    ## Utility methods ##

    @staticmethod
    def scalarize_or_tensorize(res):
        if torch.numel(res) == 1:
            return res.item()
        else:
            return Tensor(res)

    def argmax(self, axis=None, keepdim=False):
        return torch.argmax(self.data, dim=axis, keepdim=keepdim)
    
    def permute(self, dims):
        return torch.permute(self.data, dims)

    ## Basic data ##
    
    @property
    def shape(self):
        return self.data.shape
    
    ## Conversions ##

    # get_ conversion methods are O(1) and are fast. They return a different view
    # on the same memory.
    def get_ndarray(self):
        return self.data.numpy(force=False)

    # to_ conversion methods can be O(n) or otherwise be slow
    def to_ndarray(self):
        return self.data.numpy(force=True)

    ## Begin passthrough ##

    # Comparison operators
    def __lt__(self, other): return self.data.__lt__(self, other)
    def __le__(self, other): return self.data.__le__(self, other)
    def __gt__(self, other): return self.data.__gt__(self, other)
    def __ge__(self, other): return self.data.__ge__(self, other)
    def __eq__(self, other): return self.data.__eq__(self, other)
    def __ne__(self, other): return self.data.__ne__(self, other)

    # Truthiness
    # Raises error on more than one element
    def __bool__(self): return self.data.__bool__(self)

    # Arithmetic binary operators
    def __add__(self, other): return self.data.__add__(self, other)
    def __sub__(self, other): return self.data.__sub__(self, other)
    def __mul__(self, other): return self.data.__mul__(self, other)
    def __matmul__(self, other): return self.data.__matmul__(self, other)
    def __truediv__(self, other): return self.data.__truediv__(self, other)
    def __floordiv__(self, other): return self.data.__floordiv__(self, other)
    def __mod__(self, other): return self.data.__mod__(self, other)
    def __divmod__(self, other): return self.data.__divmod__(self, other)
    def __powmod__(self, other): return self.data.__powmod__(self, other)
    def __pow__(self, other, modulo=None): return self.data.__pow__(self, other, modulo)
    def __lshift__(self, other): return self.data.__lshift__(self, other)
    def __rshift__(self, other): return self.data.__rshift__(self, other)
    def __and__(self, other): return self.data.__and__(self, other)
    def __xor__(self, other): return self.data.__xor__(self, other)
    def __or__(self, other): return self.data.__or__(self, other)

    # Arithmetic unary operators
    def __neg__(self): return self.data.__neg__(self)
    def __pos__(self): return self.data.__pos__(self)
    def __abs__(self): return self.data.__abs__(self)
    def __invert__(self): return self.data.__invert__(self)

    # Arithmetic in-place
    def __iadd__(self): return self.data.__iadd__(self)
    def __isub__(self): return self.data.__isub__(self)
    def __imul__(self): return self.data.__imul__(self)
    def __imatmul__(self): return self.data.__imatmul__(self)
    def __itruediv__(self): return self.data.__itruediv__(self)
    def __ifloordiv__(self): return self.data.__ifloordiv__(self)
    def __imod__(self): return self.data.__imod__(self)
    def __idivmod__(self): return self.data.__idivmod__(self)
    def __ipowmod__(self): return self.data.__ipowmod__(self)
    def __ipow__(self, modulo=None): return self.data.__ipow__(self, modulo)
    def __ilshift__(self): return self.data.__ilshift__(self)
    def __irshift__(self): return self.data.__irshift__(self)
    def __iand__(self): return self.data.__iand__(self)
    def __ixor__(self): return self.data.__ixor__(self)
    def __ior__(self): return self.data.__ior__(self)

class Net(backends.Net):
    def __init__(self, net: torch.nn.Module):
        self.net = net

    def forwardPass(self, inputTensor: backends.Tensor) -> backends.Tensor:
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(inputTensor.get_ndarray()))
        if isinstance(out, torch.Tensor):
            return Tensor(out)
        elif isinstance(out, dict):
            return Tensor(out['out'])
    

Backend = Backend_class()