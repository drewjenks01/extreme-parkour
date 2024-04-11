
from params_proto.hyper import Sweep

import isaacgym

import os
print(os.getcwd())

from legged_gym.envs import *
from legged_gym.utils import get_args
from experiment.train_ep import train

sweep = Sweep()

def test_func():
    print("Test!")

if __name__ == '__main__':
    import jaynes
    import functools
    
    args = get_args()

    if args.both_phases:

        for i in range(1, 3):
            print(f'Training phase: {i}')

            if i == 2:
                args.exptid = args.exptid.replace('phase1', 'phase2')
                args.use_camera = True
                args.resume = True
                args.resume_path = 




    for i, kwargs in enumerate(sweep):
        verbose = True # set to True for debugging
        jaynes.config(verbose=verbose)
        train_fn = functools.partial(train, args)
        jaynes.add(train_fn)
        jaynes.execute()
    # for i, kwargs in enumerate(sweep):
    #     jaynes.config(verbose=verbose)
    #     train_fn = functools.partial(train, args)
    #     jaynes.add(train_fn)
    #     jaynes.execute()

    jaynes.listen()