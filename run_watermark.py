import sys
import subprocess

def run_watermark():
    config = {
        'script': 'postmark/watermark.py',
        'dataset': 'opengen',
        'function': 'normal',  # 注意不要打错了
        'embed_type': 'isolated',  # 注意不要打错了
        'ratio': 0.12,
        'freq': 1000,
        'alpha': 0.7,
        'up': 1.0,
        'down': 0.6,
        'freq_thresh': 1000,  # 目前效果只能拿来当baseline
        'iterate': 20,
        'iterate_type': 'v1',
        'attack': 'paraphrase',
        'm': 4,
        's': 0,
        'n': 10,
        'cache_text1': 'outputs/opengen/gpt-4_postmark-12-open.jsonl',
        # 'cache_text2': 'outputs/test_nomic_r0.12_f1000_isolated_u0.8_d0.6_i20-v1_paragraph-deepseek-v3.jsonl'
    }

    # Build the command
    cmd = [
        sys.executable,
        config['script'],
        '--dataset', config['dataset'],
        '--function', config['function'],
        '--embed_type', config['embed_type'],
        '--ratio', str(config['ratio']),
        '--freq', str(config['freq']),
        '--alpha', str(config['alpha']),
        '--up', str(config['up']),
        '--down', str(config['down']),
        '--freq_thresh', str(config['freq_thresh']),
        '--iterate', str(config['iterate']),
        '--iterate_type', config['iterate_type'],
        '--attack', config['attack'],
        '--m', str(config['m']),
        '--s', str(config['s']),
        '--n', str(config['n']),
        '--cache_text1', config['cache_text1'],
        # '--cache_text2', config['cache_text2']
    ]

    print('Executing command:')
    print(' '.join(cmd))
    print('\n')

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'\nError running command (exit code {e.returncode})')
        sys.exit(1)
    except FileNotFoundError:
        print('\nError: Python or script not found. Check paths.')
        sys.exit(1)

if __name__ == '__main__':
    run_watermark()