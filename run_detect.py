import sys
import subprocess

input_files = [  # plt默认的颜色上限只有10个
    'outputs/test_nomic_r0.06_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    'outputs/test_nomic_r0.18_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',

    # 'outputs/test_nomic_r0.12_f1000_origin_i20-v1_paraphrase-deepseek-v3.jsonl',
    'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_origin_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.6/test_nomic_r0.12_f1000_isolated_u0.8_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',

    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_i20-v1_paraphrase-deepseek-v3.jsonl',

    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.5_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.9_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.65_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.8_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/mixed/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.75_i20-v1_paraphrase-deepseek-v3.jsonl',

    # 'outputs/5.5/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_i20-v1_random_remove-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_random_remove-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_origin_i20-v1_random_remove-deepseek-v3.jsonl',
    # 'outputs/5.6/test_nomic_r0.12_f1000_isolated_u0.8_d0.6_i20-v1_random_remove-deepseek-v3.jsonl',

    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paragraph-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_origin_i20-v1_paragraph-deepseek-v3.jsonl',
    # 'outputs/5.6/test_nomic_r0.12_f1000_isolated_u0.8_d0.6_i20-v1_paragraph-deepseek-v3.jsonl',

    # 'outputs/test_nomic_r0.12_f1000_isolated_u0.9_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u0.9_d0.7_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.7_i20-v1_paraphrase-deepseek-v3.jsonl',
]
input_path = ', '.join(input_files)

cmd = [
    sys.executable,
    'postmark/detect.py',
    '--detect_type', 'normal',
    '--input_path', input_path,
    '--n', '100',
]

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f'Command failed with exit code {e.returncode}')