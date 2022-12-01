import functools, hashlib, os, random, sys
from collections import Counter

import numpy as np
import torch, torchvision
import PIL


def evalmode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = args[0]
        model.eval()
        r = func(*args, **kwargs)
        model.train()
        return r
    return wrapper


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

                
class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

        
class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_acc(logits, y):
    _, pred = torch.max(logits.data, 1)
    correct = (pred == y).sum().item()
    return correct / logits.size(0) * 100


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for y in dataset.targets:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights
    

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        if isinstance(self.underlying_dataset, torch.utils.data.TensorDataset):
            self.targets = self.underlying_dataset[self.keys][1].data.numpy()
        else:
            self.targets = [self.underlying_dataset.targets[k] for k in self.keys]
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)


    
def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
    
    
def setup(seed=0):
    ''' Reference: https://pytorch.org/docs/stable/notes/randomness.html '''
    deterministic = seed is not None
    
    if deterministic:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.enabled = True
        # if torch.__version__ >= '1.8.0':
        #     torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    else:
        device = 'cpu'
        
    print(f'Backend:')
    print(f'\tdevice: {device}')
    if device == 'cuda':
        count = torch.cuda.device_count()
        print(f'\tvisible device count: {count}')
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(count))))
        print(f'\tvisible device ID(s): {visible_devices}')
        
    return device


def pprint_dict(title, d):
    print(f'{title}:')
    for k, v in sorted(d.items()):
        print(f'\t{k}: {v}')
        
        
def list_environments():
    print('Environment:')
    print(f'\tPython: {sys.version.split(" ")[0]}')
    print(f'\tPyTorch: {torch.__version__}')
    print(f'\tTorchvision: {torchvision.__version__}')
    print(f'\tCUDA: {torch.version.cuda}')
    print(f'\tCUDNN: {torch.backends.cudnn.version()}')
    print(f'\tNumPy: {np.__version__}')
    print(f'\tPIL: {PIL.__version__}')