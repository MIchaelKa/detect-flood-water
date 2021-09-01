import numpy as np
import random
import os

import torch

def get_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('[device] There are %d GPU(s) available.' % torch.cuda.device_count())
        print('[device] We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('[device] No GPU available, using the CPU instead.')

    return device

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_num_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def display_memory(device):
    torch.cuda.empty_cache()
    memory = torch.cuda.memory_allocated(device)
    print("{:.3f} GB".format(memory / 1024 ** 3))

#
# time utils
#

import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))