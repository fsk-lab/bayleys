import copy
from dataclasses import dataclass
import numpy as np
import torch
import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from gauche.kernels.fingerprint_kernels import TanimotoKernel
from botorch.models import SingleTaskGP

from .base import Model, ModelArchitecture, TrainingConfig

from ..encoders.base import MoleculeEncoder
from ..config import get_config


KERNEL_FACTORY = {"RBF": RBFKernel, "Matern": MaternKernel, "Tanimoto": TanimotoKernel}
LIKELIHOOD_FACTORY = {"Gaussian": GaussianLikelihood}


@dataclass
class GaussianProcessArchitecture(ModelArchitecture):
    """
    Architecture configuration for Gaussian Process models.
    """
    kernel: str = "Matern"
    kernel_kwargs: dict = None
    kernel_lengthscale: bool = True
    frozen_outputscale: bool = True
    frozen_noise: bool = True

    likelihood: str = "Gaussian"

    def __post_init__(self):
        """
        Validates the Gaussian Process architecture configuration after initialization by performing the following
        checks:
          - Ensures that the specified kernel exists in the KERNEL_FACTORY.
          - Ensures that the specified likelihood exists in the LIKELIHOOD_FACTORY.
        """
        if self.kernel not in KERNEL_FACTORY:
            raise ValueError(f"Kernel '{self.kernel}' is not recognized!")
        if self.likelihood not in LIKELIHOOD_FACTORY:
            raise ValueError(f"Likelihood '{self.likelihood}' is not recognized!")
        if self.kernel_kwargs is None:
            self.kernel_kwargs = {}

    def get_kernel(self, feature_dim: int) -> Kernel:
        """
        Instantiates and returns the kernel module based on the architecture configuration. Performs the following
        adaptations:
          - If the Kernel class has `active_dims`, it is wrapped in a ScaleKernel.
          - The Kernel lengthscale prior is set to a log-normal distribution with a mean of 4.055 * sqrt(feature_dim) a
            and a variance of 20.086, which correspond to the empirical distribution of optimal lengthscales observed
            in the experiments of Hvarfner et al. (ICML 2024). The kernel lengthscale is initialized at the mode of
            this distribution, which is prior_mean / prior_var.

        Args:
            feature_dim (int): Dimensionality of the input features.

        Returns:
            An instance of the specified kernel module.
        """
        kernel_cls = KERNEL_FACTORY[self.kernel]

        if kernel_cls.has_lengthscale:

            if self.kernel_lengthscale is True:
                ard_num_dims = feature_dim
            else:
                ard_num_dims = 1

            prior_mean = torch.tensor(4.055 * feature_dim ** 0.5, dtype=torch.float32)
            prior_var = torch.tensor(20.086, dtype=torch.float32)
            lengthscale_prior = LogNormalPrior(loc=torch.log(prior_mean), scale=torch.log(prior_var).sqrt())

            kernel_obj = kernel_cls(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-5),
                **self.kernel_kwargs
            )
            kernel_obj.initialize(lengthscale=prior_mean / prior_var)  # initialize at the mode of the log-normal prior

        else:
            kernel_obj = kernel_cls(**self.kernel_kwargs)

        if hasattr(kernel_obj, "active_dims"):
            kernel_obj = ScaleKernel(kernel_obj)

        return kernel_obj

    def get_likelihood(self):
        """
        Instantiates and returns the likelihood module based on the architecture configuration.

        Returns:
            An instance of the specified likelihood module.
        """
        return LIKELIHOOD_FACTORY[self.likelihood]()


class GaussianProcessModel(Model, SingleTaskGP):
    """
    Implementation of a Gaussian Process surrogate model.
    """
    name = "GP"

    default_architecture = GaussianProcessArchitecture()
    default_training_config = TrainingConfig(learning_rate=0.05, weight_decay=0.0)

    def __init__(
            self,
            architecture: GaussianProcessArchitecture,
            training_config: TrainingConfig,
            encoder: MoleculeEncoder,
            static_embeddings: bool = True,
    ):

        likelihood = architecture.get_likelihood()
        mean_module = ZeroMean()
        covar_module = architecture.get_kernel(encoder.embedding_dim)

        SingleTaskGP.__init__(
            self,
            train_X=torch.empty(0, encoder.embedding_dim, dtype=torch.float32),
            train_Y=torch.empty(0, 1, dtype=torch.float32),
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            input_transform=None,
            outcome_transform=None
        )
        Model.__init__(self, architecture, training_config, encoder, static_embeddings)

    def train_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Trains the Gaussian Process model on the provided data, minimizing the exact marginal log likelihood.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        if self.static_embeddings is True:
            if train_embeddings is None:
                raise ValueError("train_embeddings must be provided when static_embeddings is True.")
            self.set_train_data(inputs=train_embeddings, targets=train_labels.flatten(), strict=False)
        else:
            raise NotImplementedError("Dynamic embeddings are not implemented yet.")

        device = get_config().device
        self.to(device)

        self.train(), self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        if self.architecture.frozen_outputscale is True:
            self.covar_module.raw_outputscale.requires_grad_(False)
            self.covar_module.outputscale = 1.0
        if self.architecture.frozen_noise is True:
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
            self.likelihood.noise_covar.noise = 1e-3

        optimizer = self.training_config.get_optimizer(self.parameters())

        best_loss, best_model, epochs_no_improve = torch.inf, None, 0
        losses = []

        for epoch in range(self.training_config.num_epochs):

            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

            # Performs early-stopping on the loss value rather than on the validation set, since MLL is not comparable
            # across datasets
            if self.training_config.early_stopping:
                if best_loss - loss.detach().cpu().item() > 1E-4:
                    best_loss = loss.detach().cpu().item()
                    epochs_no_improve = 0
                    best_model = copy.deepcopy(self.state_dict())
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.training_config.early_stopping_patience:
                    if best_model is None:
                        epochs_no_improve = 0
                    else:
                        self.load_state_dict(best_model)
                        break

        self.eval()

    def retrain_model(
            self,
            train_smiles: np.ndarray,
            train_labels: torch.Tensor,
            train_embeddings: torch.Tensor = None
    ):
        """
        Retrains the Gaussian Process model on the provided data. Starts from the current model parameters and
        continues training, minimizing the exact marginal log likelihood.

        Args:
            train_smiles (np.ndarray): Array of SMILES strings for training data.
            train_labels (torch.Tensor): Tensor of target labels for training data.
            train_embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for training data. Only used if
                                                       static_embeddings is True.
        """
        self.train_model(train_smiles, train_labels, train_embeddings)

    @torch.inference_mode()
    def predict(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the target labels for the provided SMILES strings using the Gaussian Process model.

        Args:
            embeddings (torch.Tensor, optional): Tensor of precomputed embeddings for prediction data.

        Returns:
            torch.Tensor: Predicted means of shape (N, 1).
            torch.Tensor: Predicted variances of shape (N, 1).
        """
        device = get_config().device

        self.to(device)
        embeddings = embeddings.to(device)

        with gpytorch.settings.fast_pred_var(True):
            posterior = self.posterior(embeddings)

        return posterior.mean.cpu(), posterior.variance.cpu()
