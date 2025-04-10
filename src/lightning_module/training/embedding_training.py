from torch import nn, cat
from lightning_module.lightning_core import LitStructureCore
from torcheval.metrics.functional import multiclass_auprc, multiclass_auroc, multiclass_confusion_matrix


class LitEmbeddingTraining(LitStructureCore):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
        return nn.functional.mse_loss(z_pred, z)

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)


class LitStoEmbeddingTraining(LitStructureCore):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def training_step(self, batch, batch_idx):
        (x, x_mask), z = batch
        z_pred = self.model(x, x_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
        # Note this differs from LitEmbedding above. Crossentropy is needed because our outputs are logits per class
        return nn.functional.cross_entropy(z_pred, z)

    def validation_step(self, batch, batch_idx):
        (x, x_mask), z = batch
        z_pred = self.model(x, x_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)

    def on_validation_epoch_end(self):
        z = self.z
        z_pred = self.z_pred
        # This is to have an indices vector as explained in
        #  https://pytorch.org/torcheval/stable/generated/torcheval.metrics.MulticlassAUROC.html
        z_nonzero_indices = z.nonzero(as_tuple=True)[1]
        pr_auc = multiclass_auprc(z_pred, z_nonzero_indices, num_classes=self.cfg.training_parameters.num_classes)
        self.log(self.PR_AUC_METRIC_NAME, pr_auc, sync_dist=True)
        # TODO Joan's code had a conditional for 'mps' device here. Is that needed?
        # if self.device.type == 'mps':
        #     roc_auc = binary_auroc(z_pred.to('cpu'), z.to('cpu'))
        # else:
        #     roc_auc = binary_auroc(z_pred, z)
        roc_auc = multiclass_auroc(z_pred, z_nonzero_indices, num_classes=self.cfg.training_parameters.num_classes)
        self.log(self.ROC_AUC_METRIC_NAME, roc_auc, sync_dist=True)
        self.log_loss(self.VALIDATION_LOSS_METRIC_NAME)
        conf_matr = multiclass_confusion_matrix(z_pred, z_nonzero_indices, self.cfg.training_parameters.num_classes)
        if self.cfg is not None and hasattr(self.logger.experiment, 'add_text'):
            # numpy gets the pure matrix without any tensor prefix or cuda suffix
            self.logger.experiment.add_text("Confusion matrix", str(conf_matr.cpu().numpy()))

def log_loss(self, step):
        # note: a specific implementation of this one is needed because for classification problem we use crossentropy
        if len(self.z) == 0:
            return
        z = self.z
        z_pred = self.z_pred
        loss = nn.functional.cross_entropy(z_pred, z)
        self.log(step, loss, sync_dist=True)
        self.reset_z()
