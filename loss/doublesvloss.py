from .loss import Loss

class DoubleSVLoss(Loss):
    def __init__(self, args, ckp):
        super(DoubleSVLoss, self).__init__(args, ckp)

    def forward(self, demosr, demos, sr, lg, sg):
        losses = []
        for i, l in enumerate(self.loss):
            if i == 0 and l['function'] is not None:
                loss = l['function']((demosr+sr)/2, sg)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()
            elif i == 1 and l['function'] is not None:
                loss = l['function'](demos, lg)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()
            elif i == 2 and l['function'] is not None:
                loss = l['function'](demosr, sr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.losslog[-1, -1] += loss_sum.item()
        
        return loss_sum