import argparse 

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

# enviroment parameters 
env_group = parser.add_argument_group("env")
env_group.add_argument('--env', type=str, default="InvertedPendulum-v4", help="name of the environment")
env_group.add_argument('--env-kwargs', type=dict, default={}, help="kwargs for the environment")
env_group.add_argument("--render", type=strToBool, default=False, help="render the environment")
# symmetry finding parameters 
sym_group = parser.add_argument_group("sym")
sym_group.add_argument("--buffer_size", type=int, default=100000, help="size of the replay buffer")
sym_group.add_argument("--save_buffer", type=strToBool, default=False, help="save the buffer")
sym_group.add_argument("--symmetry_batch_size", type=int, default=100, help="size of the symmetry batch")
sym_group.add_argument("--symmetry_num_test", type=int, default=500, help="how many times to test a transformation")
sym_group.add_argument("--delta", type=float, default=0.9, help="minimum success rate for a transformation to be considered a symmetry")
sym_group.add_argument("--epsilon", type=float, default=0.01, help="range for a transformation to be considered a symmetry")
# training parameters
train_group = parser.add_argument_group("train")

args = parser.parse_args()