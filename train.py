# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from parser import parse
import torch
import numpy as np


if __name__ == '__main__':
    args = parse()

    """ Setting seed of the RNG in all packages."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Import specified class with the experimental setup."""
    exp_args = args.experiment.split(".")
    exp_path = ".".join(exp_args[:-1])
    exp_name = exp_args[-1]
    runner_class = getattr(__import__(exp_path, fromlist=[exp_name]), exp_name)
    runner = runner_class(args)

    if not args.infer_only:
        runner.train()
    runner.infer()
