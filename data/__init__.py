from importlib import import_module

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader

class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )
        module_test = import_module('data.' +  args.data_test.lower())
        testset = getattr(module_test, args.data_test)(args, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )