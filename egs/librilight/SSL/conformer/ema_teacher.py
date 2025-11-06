import copy
import torch
from typing import Optional, Union
from torch.nn.parallel import DistributedDataParallel as DDP

class EMATeacher:
    """
    Maintains an Exponential Moving Average (EMA) of the student model parameters as the teacher model.
    The teacher model is updated as:
        teacher_param = decay * teacher_param + (1 - decay) * student_param
    The teacher model is used for self-distillation.
    """
    def __init__(self, student_model: Union[torch.nn.Module, DDP], decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = float(decay)
        # Unwrap DDP student model if necessary
        student = student_model.module if hasattr(student_model, "module") else student_model
        # Determine device
        self.device = device if device is not None else next(student.parameters()).device
        # Create a deep copy of the (unwrapped) student model as the teacher
        self.teacher_model = copy.deepcopy(student)
        self.teacher_model.eval()
        try:
            self.teacher_model.to(self.device)
        except Exception:
            pass
        # EMA teacher should not require gradients
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model: Union[torch.nn.Module, DDP]):
        """
        Update the teacher model parameters using EMA of the student model parameters.
        Handles wrapped (DDP) student models by unwrapping them first.
        """
        student = student_model.module if hasattr(student_model, "module") else student_model
        # Update parameters
        for ema_param, student_param in zip(self.teacher_model.parameters(), student.parameters()):
            ema_param.data.mul_(self.decay).add_(student_param.data.to(ema_param.device), alpha=(1.0 - self.decay))
        # Also update buffers (e.g., running_mean/running_var in BatchNorm)
        for ema_buf, student_buf in zip(self.teacher_model.buffers(), student.buffers()):
            try:
                ema_buf.copy_(student_buf.to(ema_buf.device))
            except Exception:
                # ignore mismatched buffer/device issues
                pass

    def get_teacher_model(self):
        """
        Returns the teacher model (in eval mode, on the correct device).
        """
        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        return self.teacher_model

    def state_dict(self):
        """
        Returns the state dict of the teacher model for checkpointing.
        """
        return self.teacher_model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the state dict into the teacher model (for checkpoint restore).
        """
        self.teacher_model.load_state_dict(state_dict)
