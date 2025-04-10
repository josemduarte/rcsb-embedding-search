import logging
import signal

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from config.utils import get_config_path
from dataset.sto_from_embeddings_dataset import StoDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tools import collate_single_embedding_fn
from lightning_module.training.embedding_training import LitStoEmbeddingTraining

from config.schema_config import TrainingConfig

cs = ConfigStore.instance()
cs.store(name="training_default", node=TrainingConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="training_config")
def main(cfg: TrainingConfig):
    logger.info(f"Using config file: {get_config_path()}")
    seed_everything(cfg.global_seed, workers=True)
    training_set = StoDataset(
        sto_labels_json=cfg.training_set.tm_score_file,
        embedding_path=cfg.training_set.data_path,
        # TODO extract param to configs
        subset=0.9
    )

    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=cfg.training_parameters.epoch_size if cfg.training_parameters.epoch_size > 0 else len(weights),
        replacement=True
    )
    train_dataloader = DataLoader(
        dataset=training_set,
        sampler=sampler,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.training_set.workers,
        persistent_workers=True if cfg.training_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_single_embedding_fn
    )

    validation_set = StoDataset(
        sto_labels_json=cfg.validation_set.tm_score_file,
        embedding_path=cfg.validation_set.data_path,
        # TODO extract param to configs
        subset=-0.1
    )
    validation_dataloader = DataLoader(
        dataset=validation_set,
        batch_size=cfg.validation_set.batch_size,
        num_workers=cfg.validation_set.workers,
        persistent_workers=True if cfg.validation_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_single_embedding_fn
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStoEmbeddingTraining(
        nn_model=nn_model,
        learning_rate=cfg.training_parameters.learning_rate,
        params=cfg
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStoEmbeddingTraining.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{' + LitStoEmbeddingTraining.PR_AUC_METRIC_NAME + ':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    trainer = L.Trainer(
        max_epochs=cfg.training_parameters.epochs,
        check_val_every_n_epoch=cfg.training_parameters.check_val_every_n_epoch,
        num_nodes=cfg.computing_resources.nodes,
        devices=cfg.computing_resources.devices,
        strategy=cfg.computing_resources.strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=cfg.default_root_dir
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=cfg.checkpoint
    )


if __name__ == '__main__':
    main()
