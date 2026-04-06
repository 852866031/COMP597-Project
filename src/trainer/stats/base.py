from abc import ABC, abstractmethod
import torch

_TRAINER_STATS_AUTO_DISCOVERY_IGNORE=True

class TrainerStats(ABC):
    """Abstract class used by trainers to accumulate statistics.

    This abstract class defines the interface that statistics objects used by 
    trainers must implement. Implementations that do not need specific methods 
    should implement them using the `pass` keyword to make them no-ops.

    """

    @abstractmethod
    def start_train(self) -> None:
        """Start training.

        This method should be called by trainers when starting the training loop.

        """
        pass

    @abstractmethod
    def stop_train(self) -> None:
        """Stop training.

        This method should be called by trainers when the training is done.

        """
        pass

    @abstractmethod
    def start_step(self) -> None:
        """Start a training step.
        
        This method should be called by trainers at the beginning of every 
        training iteration.
        
        """
        pass

    @abstractmethod
    def stop_step(self) -> None:
        """Stop a training step.

        This method should be called by trainers at the end of every training 
        iteration.
        
        """
        pass

    @abstractmethod
    def start_forward(self) -> None:
        """Start the forward pass.

        This method should be called by trainers at the beginning of every 
        forward pass.

        """
        pass

    @abstractmethod
    def stop_forward(self) -> None:
        """Stop the forward pass.

        This method should be called by trainers at the end of every forward 
        pass.

        """
        pass

    @abstractmethod
    def log_loss(self, loss: torch.Tensor) -> None:
        """Logs the loss of the current step by passing it to the stats.

        """
        pass

    @abstractmethod
    def start_backward(self) -> None:
        """Start the backward pass.

        This method should be called by trainers at the start of every backward 
        pass.

        """
        pass

    @abstractmethod
    def stop_backward(self) -> None:
        """Stop the backward pass

        This method should be called by trainers at the end of every backward 
        pass.

        """
        pass

    @abstractmethod
    def start_optimizer_step(self) -> None:
        """Start the optimizer step.

        This method should be called by trainers at the start of the optimizer 
        step.

        """
        pass

    @abstractmethod
    def stop_optimizer_step(self) -> None:
        """Stop the optimizer step.

        This method should be called by trainers at the end of the optimizer 
        step.

        """
        pass

    @abstractmethod
    def start_save_checkpoint(self) -> None:
        """Start checkpointing.

        This method should be called by trainers when they initiate a 
        checkpointing step.

        """
        pass

    @abstractmethod
    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing.

        This method should be called by trainers at the end of a checkpointing 
        step.

        """
        pass

    @abstractmethod
    def log_step(self) -> None:
        """Logs information about the previous step.

        This method should be called after the `stop_step`. It should log 
        information about the previous training step.

        """
        pass

    @abstractmethod
    def log_stats(self) -> None:
        """Logs information about the data accumulated so far.

        This method should be called to log information about the data 
        accumulated. Typically, this is likely to be called after `stop_train`, 
        but implementations could want to leverage this differently.

        """
        pass

