import torch
import lightning.pytorch as pl
from typing import Optional
from self_supervision.models.lightning_modules.cellnet_autoencoder import MLPClassifier
from self_supervision.estimator.cellnet import EstimatorAutoEncoder
from self_supervision.trainer.classifier.cellnet_mlp import update_weights


def load_model(
    model_dir: str,
    estim: pl.LightningModule,
    supervised_subset: Optional[int] = None,
) -> pl.LightningModule:
    """
    Load a model from a given directory.

    Args:
        model_dir (str): The directory path where the model is saved.
        estim (pl.LightningModule): The estimator object.
        supervised_subset (Optional[int], optional): The number of supervised samples to use. Defaults to None.

    Returns:
        pl.LightningModule: The loaded model.

    Raises:
        ValueError: If the model directory does not contain a valid model.
    """

    if supervised_subset:
        print("Loading model with supervised_subset: ", supervised_subset)

    if "final_model" in model_dir:
        return MLPClassifier.load_from_checkpoint(
            model_dir,
            **estim.get_fixed_clf_params(),
            units=[512, 512, 256, 256, 64],
            supervised_subset=supervised_subset,
        )
    elif "pretext_model" in model_dir:
        # If estim has no attribute model yet, initialize it
        if not hasattr(estim, "model"):
            print("Initializing model...")
            estim.init_model(
                model_type="mlp_clf",
                model_kwargs={
                    "learning_rate": 1e-3,
                    "weight_decay": 0.1,
                    "lr_scheduler": torch.optim.lr_scheduler.StepLR,
                    "dropout": 0.1,
                    "lr_scheduler_kwargs": {
                        "step_size": 2,
                        "gamma": 0.9,
                        "verbose": True,
                    },
                    "units": [512, 512, 256, 256, 64],
                    "supervised_subset": supervised_subset,
                },
            )
        final_dict = update_weights(model_dir, estim)
        estim.model.load_state_dict(final_dict)
        # only change the supervised_subset attribute of the estim.model
        estim.model.supervised_subset = supervised_subset
        print("Supervised subset: ", estim.model.supervised_subset)
        return estim.model
    else:
        raise ValueError("Model directory does not contain a valid model.")


class LightningWrapper(pl.LightningModule):
    def __init__(self, base_model):
        """
        Initializes the LightningWrapper class.

        Args:
            base_model: The base model to be wrapped.
        """
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.base_model(x)

    def predict_step(self, *args, **kwargs):
        """
        Performs a prediction step using the base model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Prediction result.
        """
        return self.base_model.predict_step(*args, **kwargs)


def load_and_wrap_model(model_dir, estim, supervised_subset):
    """
    Load and wrap a model.

    Args:
        model_dir (str): The directory path where the model is stored.
        estim (str): The estimation method.
        supervised_subset (bool): Whether to use a subset of the data for supervised training.

    Returns:
        wrapped_model: The wrapped model.

    """
    model = load_model(model_dir, estim, supervised_subset)
    wrapped_model = LightningWrapper(model)
    return wrapped_model


def prepare_estim(estim, wrapped_model, batch_size):
    """
    Prepares the Estimator object for prediction by re-initializing the estim.model and assigning the wrapped model.

    Args:
        estim (EstimatorAutoEncoder): The Estimator object.
        wrapped_model (torch.nn.Module): The wrapped model for prediction.
        batch_size (int): The batch size for prediction.

    Returns:
        EstimatorAutoEncoder: The updated Estimator object.
    """
    # Re-initialize the estim.model to avoid nesting somehow
    # Assign the wrapped model to estim for prediction
    estim = EstimatorAutoEncoder(data_path=estim.data_path, hvg=estim.hvg)
    estim.init_datamodule(batch_size=batch_size)
    # Assign the wrapped model to estim for prediction
    estim.model = None
    estim.model = wrapped_model
    estim.trainer = pl.Trainer(logger=[], accelerator="gpu", devices=1)

    return estim
