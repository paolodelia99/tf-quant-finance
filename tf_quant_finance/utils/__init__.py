# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities module."""

from tf_quant_finance.utils.dataclass import dataclass
from tf_quant_finance.utils.shape_utils import broadcast_common_batch_shape
from tf_quant_finance.utils.shape_utils import broadcast_tensors
from tf_quant_finance.utils.shape_utils import common_shape
from tf_quant_finance.utils.shape_utils import get_shape
from tf_quant_finance.utils.tf_functions import iterate_nested

from tensorflow.python.util.all_util import (
    remove_undocumented,
)  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    "dataclass",
    "broadcast_common_batch_shape",
    "broadcast_tensors",
    "common_shape",
    "get_shape",
    "iterate_nested",
]

remove_undocumented(__name__, _allowed_symbols)
