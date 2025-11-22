import math

class StableLRScheduler:
    def __init__(self, optimizer, initial_lr, warmup_epochs, max_epochs, min_lr=1e-6, max_lr_multiplier=1.0):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = initial_lr * max_lr_multiplier
        self.current_epoch = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr * 0.1

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr * (0.1 + 0.9 * self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        lr = max(self.min_lr, min(lr, self.max_lr))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
