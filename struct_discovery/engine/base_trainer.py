"""Implements a get base trainer."""
import torch
from torch.utils.tensorboard import SummaryWriter

from ignite.utils import convert_tensor
from ignite.engine import Events, create_supervised_evaluator
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver

from struct_discovery.solver.build import build_optimizer
from struct_discovery.solver.hypergrad import implicit_function


from itertools import repeat


class BaseTrainer(object):
    def __init__(self, cfg):
        """Builds an experiment from config."""
        self.cfg = cfg
        # Build data
        self.train_loader = self.build_train_loader(cfg)
        self.val_loader = self.build_val_loader(cfg)

        def repeater(data_loader):
            for loader in repeat(data_loader):
                for data in loader:
                    yield data

        self.hyper_train_loader_iter = iter(repeater(self.train_loader))
        self.hyper_val_loader_iter = iter(repeater(self.val_loader))
        self.test_loader = self.build_test_loader(cfg)

        # Build model
        self.model = self.build_model(cfg)
        self.optimizer, self.hyper_optimizer = self.build_optimizers(
            cfg, self.model)
        self.device = None
        self.non_blocking = False
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        # Build trainer
        def _update(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = self.prepare_batch(
                batch, device=self.device, non_blocking=self.non_blocking)
            y_pred = self.model(x)
            total_loss = self.model.total_loss(y_pred, y)
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item()

        deterministic = True
        trainer = Engine(
            _update) if not deterministic else DeterministicEngine(_update)

        # Add outter loop support.
        @trainer.on(Events.ITERATION_COMPLETED(every=cfg.SOLVER.UPDATE_PERIOD_HYPER))
        def perform_outter_loop(engine):
            self.model.train()
            if cfg.SOLVER.IMPLICIT_GRADIENT_METHOD in ['EXACT', 'IDENTITY', 'NEUMANN', 'CG']:
                # Perform a hypergradient update.
                self.hyper_optimizer.zero_grad()
                val_batch = next(self.hyper_val_loader_iter)
                val_data = self.prepare_batch(
                    val_batch, device=self.device, non_blocking=self.non_blocking)
                train_batch = next(self.hyper_train_loader_iter)
                train_data = self.prepare_batch(
                    val_batch, device=self.device, non_blocking=self.non_blocking)
                hyper_grad = implicit_function.compute_hypergrad(train_data, val_data,
                                                                 self.model,
                                                                 method=self.cfg.SOLVER.IMPLICIT_GRADIENT_METHOD, cfg=cfg)
                with torch.no_grad():
                    bidx = 0
                    for mm in self.model.hyper_parameters():
                        mm_size = mm.nelement()
                        eidx = bidx + mm_size
                        mm.grad = torch.reshape(
                            hyper_grad[bidx:eidx, :], mm.shape).clone()
                    self.hyper_optimizer.step()
                    print(list(self.model.hyper_parameters()))
        # Build Evaluator
        evaluator = self.build_evaluator(cfg, self.model, self.device)
        # Add logging support.
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR + '/logs/')

        @trainer.on(Events.ITERATION_COMPLETED(every=1))
        def log_training_loss(engine):
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.5f}"
                "".format(engine.state.epoch, engine.state.iteration,
                          len(self.train_loader), engine.state.output)
            )
            writer.add_scalar("training/loss", engine.state.output,
                              engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.val_loader)
            metrics = evaluator.state.metrics
            total_loss = metrics["total_val_loss"]
            print(
                "Validation Results - Epoch: {}  Avg loss: {:.5f}".format(
                    engine.state.epoch, total_loss
                )
            )
            writer.add_scalar("valdation/avg_loss",
                              total_loss, engine.state.epoch)
            engine.state.metrics["total_val_loss"] = total_loss

        # Add Checkpoint support after every epoch.
        output_path = cfg.OUTPUT_DIR + '/saved_checkpoints/'
        if output_path is not None:
            save_handler = DiskSaver(dirname=output_path, require_empty=False)
        to_save = {"trainer": trainer, "model": self.model,
                   "param_optimizer": self.optimizer, "hyper_optimizer": self.hyper_optimizer}
        checkpoint_handler = Checkpoint(
            to_save, save_handler, filename_prefix="training", n_saved=5)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

        # Add checkpoint support to save best validation.
        def score_function(engine):
            return -1*engine.state.metrics['total_val_loss']
        to_save_best = {'model': self.model}
        best_handler = Checkpoint(to_save_best, save_handler, n_saved=2,
                                  filename_prefix='best', score_function=score_function, score_name="val_loss")
        trainer.add_event_handler(Events.EPOCH_COMPLETED, best_handler)

        self.trainer = trainer
        self.writer = writer

    def train(self,):
        num_epoch = self.cfg.SOLVER.MAX_ITER//len(self.train_loader)
        self.trainer.run(self.train_loader, num_epoch)

    def test(self,):
        pass

    @classmethod
    def prepare_batch(cls, batch, device, non_blocking=False):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    @classmethod
    def build_model(cls, cfg):
        raise NotImplementedError()

    @classmethod
    def build_optimizers(cls, cfg, model):
        param_optimizer = cls.build_param_optimizer(cfg, model)
        hyper_optimizer = cls.build_hyper_optimizer(cfg, model)
        return param_optimizer, hyper_optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        raise NotImplementedError()

    @classmethod
    def build_val_loader(cls, cfg):
        raise NotImplementedError()

    @classmethod
    def build_test_loader(cls, cfg):
        raise NotImplementedError()

    @classmethod
    def build_param_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model.model_parameters(),
                               cfg.SOLVER.BASE_LR_PARAM,
                               cfg.SOLVER.NAME_PARAM)

    @classmethod
    def build_hyper_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model.hyper_parameters(),
                               cfg.SOLVER.BASE_LR_PARAM,
                               cfg.SOLVER.NAME_HYPER)

    @classmethod
    def build_evaluator(cls, cfg, model, device):
        val_metrics = {"total_val_loss": Loss(model.total_val_loss)}
        evaluator = create_supervised_evaluator(
            model, val_metrics, device=device)
        return evaluator
