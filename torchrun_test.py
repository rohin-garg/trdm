import os
import torch.distributed as dist
print('hello from pid', os.getpid(), 'rank', os.environ.get('RANK'), 'world', os.environ.get('WORLD_SIZE'))
dist.init_process_group('gloo')
print('init done', os.getpid())
dist.destroy_process_group()
