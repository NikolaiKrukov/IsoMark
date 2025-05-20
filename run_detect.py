import sys
import subprocess

input_files = [  # 注意plt颜色上限
    # 5.5-r-isolated
    # 'outputs/5.5r/test_nomic_r0.06_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.5r/test_nomic_r0.18_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',

    # main_500 & G (注意调整参数)
    # 'outputs/test_nomic_r0.12_f1000_origin_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_origin_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',

    # main_100 (注意调整参数)
    'outputs/test_nomic_r0.12_f1000_origin_paraphrase-deepseek-v3.jsonl',
    'outputs/test_nomic_r0.12_f1000_origin_i20-v1_paragraph-deepseek-v3.jsonl',
    'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_paraphrase-deepseek-v3.jsonl',
    'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_paraphrase-deepseek-v3.jsonl',
    'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_i20-v1_paraphrase-deepseek-v3.jsonl',
    
    # 5.5a
    # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.5_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.65_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_i20-v1_paraphrase-deepseek-v3.jsonl',
    # # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.75_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.8_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.9_i20-v1_paraphrase-deepseek-v3.jsonl',
    # # 'outputs/5.5a/test_nomic_r0.12_f1000_mixed_u1.0_d0.6_a0.7_paraphrase-deepseek-v3.jsonl',

    # 5.4
    # 'outputs/5.4/test_nomic_r0.12_f1000_origin_i20-v1_paragraph-deepseek-v3.jsonl',
    # 'outputs/5.4/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paragraph-deepseek-v3.jsonl',
    # 'outputs/5.4/test_nomic_r0.12_f1000_isolated_u0.8_d0.6_i20-v1_paragraph-deepseek-v3.jsonl',

    # 5.5-du
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.4/test_nomic_r0.12_f1000_isolated_u0.8_d0.6_i20-v1_paragraph-deepseek-v3.jsonl',
    # # 'outputs/5.5du/test_nomic_r0.12_f1000_isolated_u0.9_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # # 'outputs/5.5du/test_nomic_r0.12_f1000_isolated_u0.9_d0.7_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/5.5du/test_nomic_r0.12_f1000_isolated_u1.0_d0.7_i20-v1_paraphrase-deepseek-v3.jsonl',

    # F
    # 'outputs/test_nomic_r0.12_f1000_origin_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/test_nomic_r0.12_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl',
    # 'outputs/F/test_nomic_r0.12_f1000_random_i20-v1_paraphrase-deepseek-v3.jsonl',
]
input_path = ', '.join(input_files)

cmd = [
    sys.executable,
    'postmark/detect.py',
    '--detect_type', 'normal',
    '--input_path', input_path,
    '--n', '500',
]
# cmd1 = [
#     sys.executable,
#     'postmark/detect.py',
#     '--detect_type', 'normal_rotate',
#     '--input_path', input_path,
#     '--n', '500',
# ]
# cmd2 = [
#     sys.executable,
#     'postmark/detect.py',
#     '--detect_type', 'limit',
#     '--input_path', input_path,
#     '--n', '500',
# ]
# cmd3 = [
#     sys.executable,
#     'postmark/detect.py',
#     '--detect_type', 'limit_rotate',
#     '--input_path', input_path,
#     '--n', '500',
# ]

try:
    subprocess.run(cmd, check=True)  # check=True: 如果子进程的返回码（return code）不是 0，则抛出 subprocess.CalledProcessError 异常。 
    # subprocess.run(cmd1, check=True)
    # subprocess.run(cmd2, check=True)
    # subprocess.run(cmd3, check=True)
except subprocess.CalledProcessError as e:
    print(f'Command failed with exit code {e.returncode}')