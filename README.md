# Alexa End-to-End SLU

This setup allows to train end-to-end neural models for spoken language understanding (SLU).
It uses either the Snips SLU or the Fluent Speech dataset (FSC).
This framework is built using pytorch with torchaudio and the transformer package from HuggingFace.
We tested using pytorch 1.5.0 and torchaudio 0.5.0.

## Installation and data preparation

To install the required python packages, please run `pip install -r requirements.txt`. This setup uses the `bert-base-cased` model.
Typically, the model will be downloaded (and cached) automatically when running the training for the first time.
In case you want to download the model explicitly, you can run the `download_bert.py` script from the `dataprep/` directory,
e.g. `python download_bert.py bert-base-cased ./models/bert-base-cased`

This setup expects the FluentSpeechCommands dataset to reside under `fluent/` and the Snips SLU dataset under `snips_slu/`.
Please download and extract the datasets to these locations (or create a symlink).
To preprocess the Snips dataset, please run `prepare_snips.py` (located in the `dataprep/` directory) from within the `snips_slu/` folder dataset.
This will generate additional files within the `snips_slu/` folder required by the dataloader.

## Running experiments

Core to running experiments is the `train.py` script.
When called without any parameters, it will train a model using triplet loss on the FSC dataset.
The default location for saving intermediate results is the `runs/` directory.
In case it does not yet exist, it will be created.

To customize the experiments, several command line options are available (for a full list, please refer to `parser.py`):

* --dataset (The dataset to use, e.g. `fsc`)
* --experiment (The experiment class to run, e.g. `experiments.experiment_triplet.ExperimentRunnerTriplet`)
* --scheduler (Learning rate scheduler)
* --output-prefix (The prefix under which the training artifacts are being stored.)
* --bert-model-name (Name or path of pretrained BERT model to use)
* --infer-only (Only run inference on the saved model)

## Example runs

To check if everything is installed correctly, training a model with either Snips SLU or Fluent Speech Commands should produce the following results:

### Fluent Speech Commands

`python train.py --dataset fsc`

Final test acc = 0.9565, test loss = 0.5085

### Snips SLU

`python train.py --dataset snips`

Final test acc = 0.6988, test loss = 2.2471


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
