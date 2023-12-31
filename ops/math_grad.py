'''
 This file is created based on ops.math_grad.py in TF2
'''


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradients for operators defined in math_ops.py."""
import numpy as np

from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops

import tensorflow as tf


def _MatMulGradAgainstFirstOnly(op, grad):
    """Gradient for MatMul, only for the first input."""
    t_a = op.get_attr("transpose_a")
    t_b = op.get_attr("transpose_b")
    b = math_ops.conj(op.inputs[1])
    if not t_a and not t_b:
        grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
    elif not t_a and t_b:
        grad_a = gen_math_ops.mat_mul(grad, b)
    elif t_a and not t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True)
    elif t_a and t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True)
    return grad_a, None


def _MatMulGradAgainstSecondOnly(op, grad):
    """Gradient for MatMul, only for the second input."""
    t_a = op.get_attr("transpose_a")
    t_b = op.get_attr("transpose_b")
    a = math_ops.conj(op.inputs[0])
    if not t_a and not t_b:
        grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    elif not t_a and t_b:
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
    elif t_a and not t_b:
        grad_b = gen_math_ops.mat_mul(a, grad)
    elif t_a and t_b:
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True)
    return None, grad_b

