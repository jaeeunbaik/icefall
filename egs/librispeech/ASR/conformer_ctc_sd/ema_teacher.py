import copy
import torch

class EMATeacher:
    """
    Maintains an Exponential Moving Average (EMA) of the student model parameters as the teacher model.
    The teacher model is updated as:
        teacher_param = decay * teacher_param + (1 - decay) * student_param
    The teacher model is used for self-distillation.
    """
    def __init__(self, student_model, decay=0.999, device=None):
        self.decay = decay
        self.device = device if device is not None else next(student_model.parameters()).device
        # Create a deep copy of the student model as the teacher
        self.teacher_model = copy.deepcopy(student_model)
        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        # EMA should not require gradients
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model):
        """
        Update the teacher model parameters using EMA of the student model parameters.
        """
        for ema_param, student_param in zip(self.teacher_model.parameters(), student_model.parameters()):
            ema_param.data.mul_(self.decay).add_(student_param.data, alpha=1.0 - self.decay)
        # Also update buffers (e.g., running_mean/running_var in BatchNorm)
        for ema_buf, student_buf in zip(self.teacher_model.buffers(), student_model.buffers()):
            ema_buf.copy_(student_buf)

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
