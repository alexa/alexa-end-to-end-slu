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

import argparse

def parse():
    parser = argparse.ArgumentParser(description="Run an experiment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Seed for the RNG")

    parser.add_argument("--dataset",
                        choices=['fsc', 'snips'],
                        default='fsc',
                        help="The dataset to use")

    parser.add_argument("--experiment",
                        default="experiments.experiment_triplet.ExperimentRunnerTriplet",
                        help="Experiment to run")

    parser.add_argument("-lr", "--learning-rate",
                        type=float,
                        default=1e-3,
                        help="Learning rate of the acoustic encoder")

    parser.add_argument("-lr-bert", "--learning-rate-bert",
                        type=float,
                        default=2e-5,
                        help="Learning rate of the BERT model")

    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Batch size")

    parser.add_argument("--num-epochs",
                        type=int,
                        default=20,
                        help="Number of epochs to the train the model`")

    parser.add_argument("--print-every",
                        type=int,
                        default=20,
                        help="Print model stats every n steps")

    parser.add_argument("--val-every",
                        type=int,
                        default=50,
                        help="Validate model every n steps`")

    parser.add_argument("--save_every",
                        type=int,
                        default=500,
                        help="Save model every n steps`")

    parser.add_argument("--infer-only",
                        action='store_true',
                        help="Only run inference on the saved model")

    parser.add_argument("--visualize",
                        action='store_true',
                        help="User tensorboard for visualizing training curves")

    parser.add_argument("--distributed",
                        action='store_true',
                        help="Use multiple GPUs for training")

    parser.add_argument("--bert-random-init",
                        action='store_true',
                        help="Use a randomly initialized BERT model")

    parser.add_argument("--num-workers",
                        type=int,
                        default=7,
                        help="Number of concurrent dataloader threads")

    parser.add_argument("-m", "--margin",
                        type=float,
                        default=1.0,
                        help="Margin for embedding losses")

    parser.add_argument("--g-steps",
                        type=int,
                        default=1,
                        help="Number of generator steps per batch")

    parser.add_argument("--d-steps",
                        type=int,
                        default=5,
                        help="Number of discriminator steps per batch")

    parser.add_argument("--scheduler",
                        choices=['plateau', 'cycle', 'none'],
                        default='none',
                        help="Learning rate scheduler")

    parser.add_argument("--bert-model-name",
                        default="bert-base-cased",
                        help="Name or path of pretrained BERT model to use")

    parser.add_argument("--num-enc-layers",
                        default=3,
                        type=int,
                        help="Number of encoder LSTM layers")

    parser.add_argument("--enc-dim",
                        default=512,
                        type=int,
                        help="Hidden dimension of encoder LSTM")

    parser.add_argument("--weight-text",
                        default=1,
                        type=float,
                        help="Weight of the text classification loss for joint models")

    parser.add_argument("--weight-embedding",
                        default=1,
                        type=float,
                        help="Weight of the embedding loss for joint models")

    parser.add_argument("--weight-adversarial",
                        default=1,
                        type=float,
                        help="Weight of the adversarial loss for joint models")

    parser.add_argument("--model-dir",
                        default='./',
                        type=str,
                        help="Directory to store the trained model checkpoints.")

    parser.add_argument("--eval-checkpoint-path",
                        default=None,
                        type=str,
                        help="Checkpoint path to be used for testing.")

    args = parser.parse_args()
    return args
