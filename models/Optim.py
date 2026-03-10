import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup


class Optim:
    def __init__(self, lr, T, max_grad_value, weight_decay, epochs=None, batches=None):
        self.lr = lr
        self.T =  T
        self.min_lr = lr*0.5
        self.max_grad_value = max_grad_value
        self.weight_decay = weight_decay
        self.params = None
        self.optimizer = None
        self.epochs = epochs
        self.batches=batches

    def set_parameters(self, params, name):
        self.params = list(params)
        if name == "sgd":
            self.optimizer = optim.SGD(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adam":
            self.optimizer = optim.Adam(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adamw":
            self.optimizer = optim.AdamW(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )

    def get_scheduler(self, sch):
        print(f"DEBUG: Scheduler name passed: {sch}")

        print("Using Scheduler")
        if sch == "reduceLR":
            sched = lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=self.T)
        elif sch == "cyclicLR":
            sched = lr_scheduler.CyclicLR(self.optimizer, base_lr=self.min_lr, max_lr=self.lr, step_size_up=1, step_size_down=self.T, mode='triangular2')
        elif sch == "cosineLR":
            sched = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.T, T_mult=2, eta_min=0.0)
        elif sch == "cosineLR_linearWarmUp2":
            warmup_sched = lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=self.T)
            main_sched = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.T, T_mult=2, eta_min=self.min_lr)
            sched = lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[self.T]
            )
        elif sch == "cosineLR_linearWarmUp":
            sched = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.T*self.batches, num_training_steps=self.epochs*self.batches)
        elif sch == "multistepLR":
            sched = lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.T], gamma=0.1)
        elif sch == "None":
            return None
        else:
            raise NotImplementedError(f"Unavailable scheduler: {sch}")
        return sched

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.max_grad_value != -1:
            clip_grad_value_(self.params, self.max_grad_value)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        """현재 학습률 반환"""
        return self.optimizer.param_groups[0]['lr']