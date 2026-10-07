import yaml
from argparse import ArgumentParser
from scripts import generate_burgers, generate_darcy, generate_poisson, generate_helmholtz, generate_ns_nonbounded, generate_ns_bounded

if __name__ == "__main__":
    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--offset', type=int, default=None, help='Override data.offset.')
    parser.add_argument('--seed', type=int, default=None, help='Override generate.seed.')
    parser.add_argument('--iterations', type=int, default=None, help='Override test.iterations (K).')
    parser.add_argument('--run-id', type=str, default=None, help='Override generate.run_id.')
    options = parser.parse_args()
    config_path = options.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if options.offset is not None:
        config.setdefault('data', {})['offset'] = options.offset
    if options.seed is not None:
        config.setdefault('generate', {})['seed'] = options.seed
    if options.iterations is not None:
        config.setdefault('test', {})['iterations'] = options.iterations
    if options.run_id is not None:
        config.setdefault('generate', {})['run_id'] = options.run_id
    name = config['data']['name']
    if name == 'Burgers':
        print('Solving Burgers equation...')
        generate_burgers(config)
    elif name == 'Darcy':
        print('Solving Darcy Flow equation...')
        generate_darcy(config)
    elif name == 'Poisson':
        print('Solving Poisson equation...')
        generate_poisson(config)
    elif name == 'Helmholtz':
        print('Solving Helmholtz equation...')
        generate_helmholtz(config)
    elif name == 'NS-NonBounded':
        print('Solving non-bounded NS equation...')
        generate_ns_nonbounded(config)
    elif name == 'NS-Bounded':
        print('Solving bounded NS equation...')
        generate_ns_bounded(config)
    else:
        print('PDE not found')
        exit(1)