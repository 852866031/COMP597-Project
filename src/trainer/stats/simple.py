import logging
import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.utils as utils
import torch

logger = logging.getLogger(__name__)

trainer_stats_name="simple"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to simple trainer stats. Using default PyTorch device")
        device = torch.get_default_device()
    return SimpleTrainerStats(device=device)

class SimpleTrainerStats(base.TrainerStats):
    """Provides simple timing measurements of training.

    This class measures the time used by the training steps, forward passes, 
    backward passes and optimizer steps.

    Parameters
    ----------
    device
        The PyTorch device used for training. The asynchronous nature of CUDA 
        implementations means the Python code of a pass might complete before 
        the GPU is done executing the tasks. As such, this device is used to 
        synchronize on the CUDA stream to which the executions are issued.

    Attributes
    ----------
    device : torch.device
        The PyTorch device as provided to the constructor.
    step_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each training step.
    forward_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each forward pass.
    backward_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each backward pass.
    optimizer_step_stats : src.trainer.stats.RunningTimer
        Timer used to track the time used by each optimizer step.

    Notes
    -----
        This is should only be used when training is done on a CUDA device. It 
        will fail otherwise. Moreover, if the training is not done on the 
        default stream of the device, the measurements will be unreliable as 
        synchronization is only done on the default stream.

    """

    def __init__(self, device : torch.device) -> None:
        super().__init__()
        self.device = device
        self.step_stats = utils.RunningTimer()
        self.forward_stats = utils.RunningTimer()
        self.backward_stats = utils.RunningTimer()
        self.optimizer_step_stats = utils.RunningTimer()
        self.save_checkpoint_stats = utils.RunningTimer()

    def start_train(self) -> None:
        pass

    def stop_train(self) -> None:
        pass

    def start_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.step_stats.start()

    def stop_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.step_stats.stop()

    def start_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.optimizer_step_stats.start()

    def stop_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.optimizer_step_stats.stop()

    def start_forward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.forward_stats.start()

    def stop_forward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.forward_stats.stop()

    def start_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.backward_stats.start()

    def stop_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.backward_stats.stop()

    def start_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)
        self.save_checkpoint_stats.start()

    def stop_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)
        self.save_checkpoint_stats.stop()

    def log_step(self) -> None:
        """Log the previous step's time measurements.

        This will print the measured time of the previous step, its forward 
        pass, backward pass and optimizer step. All the measurements are in 
        milliseconds.

        """
        print(f"step {self.step_stats.get_last() / 1000000} -- forward {self.forward_stats.get_last() / 1000000} -- backward {self.backward_stats.get_last() / 1000000} -- optimizer step {self.optimizer_step_stats.get_last() / 1000000}")

    def log_stats(self) -> None:
        """Log basic statistics on the time measurements.

        This will print the average time of each step, each forward pass, each 
        backward pass and each optimizer step. Then it prints a breakdown for 
        each of those. All measurements are in milliseconds.

        """
        print(f"AVG : step {self.step_stats.get_average() / 1000000} -- forward {self.forward_stats.get_average() / 1000000} -- backward {self.backward_stats.get_average() / 1000000} -- optimizer step {self.optimizer_step_stats.get_average() / 1000000}")
        print("###############        Step        ###############")
        self.step_stats.log_analysis()
        print("###############      FORWARD       ###############")
        self.forward_stats.log_analysis()
        print("###############      BACKWARD      ###############")
        self.backward_stats.log_analysis()
        print("###############   OPTIMIZER STEP   ###############")
        self.optimizer_step_stats.log_analysis()
        # NOTE: (greta) commented out for now - not using checkpointing stats
        # print("###############   CHECKPOINTING    #################")
        # self.save_checkpoint_stats.log_analysis()

    def log_loss(self, loss : torch.Tensor) -> None:
        pass

