import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import json
import numpy as np
from embed import nlp
from detect_utils import evaluate, evaluate_replace
import matplotlib.pyplot as plt
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, required=True, help='Path to the JSONL data to be evaluated (separated by comma).')
parser.add_argument('--thresh', type=float, help='Cosine similarity threshold to use for computing word presence. 0 means to iterate the thresholds to find the best.', default=0)
parser.add_argument('--detect_type', type=str, help='normal: Origin detect method. limit: Add POS limit for text to be detected. cluster: Isolated clustering method.', choices=['normal', 'normal_rotate', 'limit', 'limit_rotate'], default='normal')

parser.add_argument('--output_path', type=str, help='JSONL path to save the scores along with the input data.')

parser.add_argument('--m', type=int, help='Number of thread to use.', default=1)
parser.add_argument('--s', type=int, help='LHS of instances to input.', default=0)
parser.add_argument('--n', type=int, help='RHS of instances to input.', default=0)

args = parser.parse_args()
print(args)

def save(data, attack, text1s, list1s, text2s, list2s, text3s, list3s, score1s, score2s, score3s=None):
    if args.output_path:
        with open(args.output_path, 'w') as f:
            for i in range(len(data)):
                record = {
                    'text1': text1s[i],
                    'list1': list1s[i],
                    'score1': score1s[i],
                    'text2': text2s[i],
                    'list2': list2s[i],
                    'score2': score2s[i]
                }
                if attack:
                    record.update({
                        'text3': text3s[i],
                        'list3': list3s[i],
                        'score3': score3s[i]
                    })
                f.write(json.dumps(record) + '\n')

def process_input(input_path, thresh=args.thresh, s=args.s, n=args.n):
    with open(input_path, 'r') as f:
        data = []
        for i, line in enumerate(f):
            if i < s:
                continue
            if n and i >= n:
                break
            data.append(json.loads(line))
    if not data:
        raise ValueError(f'Error: The data is empty.')
    
    attack = 'text3' in data[0]  # whether the input data contains paraphrased text

    score1s, score2s, score3s = [], [], []

    text1s = [dd['text1'] for dd in data]
    list1s = [dd['list1'] for dd in data]

    text2s = [dd['text2'] for dd in data]
    list2s = [dd['list2'] for dd in data]

    _text1s = list(nlp.pipe(text1s))
    _text2s = list(nlp.pipe(text2s))

    if attack:
        text3s = [dd['text3'] for dd in data]
        list3s = [dd['list3'] for dd in data]

        _text3s = list(nlp.pipe(text3s))
        evaluate_replace(list1s, _text2s, _text3s)

    if 'score1' in data[0]:
        score1s = [dd['score1'] for dd in data]
        score2s = [dd['score2'] for dd in data]
        if attack:
            score3s = [dd['score3'] for dd in data]

    target_fpr = 0.01
    
    if thresh:
        if attack:
            tpr_fpr, roc_auc, tpr_fpr_, roc_auc_, score1s, score2s, score3s, p_value, fp_rate, max_sliding_std = evaluate(attack, _text1s, _text2s, _text3s, list1s, list2s, list3s, thresh, target_fpr=target_fpr, detect_type=args.detect_type)
            save(data, attack, text1s, list1s, text2s, list2s, text3s, list3s, score1s, score2s, score3s)
        else:
            tpr_fpr, roc_auc, score1s, score2s = evaluate(attack, _text1s, _text2s, None, list1s, list2s, None, thresh, target_fpr=target_fpr, detect_type=args.detect_type)
            save(data, attack, text1s, list1s, text2s, list2s, None, None, score1s, score2s)

        return [None] * 6
    else:  # 暂时没有save功能
        threshs = np.arange(0.3, 1.0 + 1e-6, 0.05)
        tpr_fprs = []
        roc_aucs = []
        tpr_fprs_A = []
        roc_aucs_A = []
        for thresh in threshs:
            if attack:
                tpr_fpr, roc_auc, tpr_fpr_, roc_auc_, score1s, score2s, score3s, p_value, fp_rate, max_sliding_std = evaluate(attack, _text1s, _text2s, _text3s, list1s, list2s, list3s, thresh, target_fpr=target_fpr, detect_type=args.detect_type)
                tpr_fprs.append(tpr_fpr)
                roc_aucs.append(roc_auc)
                tpr_fprs_A.append(tpr_fpr_)
                roc_aucs_A.append(roc_auc_)
            else:
                tpr_fpr, roc_auc, score1s, score2s = evaluate(attack, _text1s, _text2s, None, list1s, list2s, None, thresh, target_fpr=target_fpr, detect_type=args.detect_type)
                tpr_fprs.append(tpr_fpr)
                roc_aucs.append(roc_auc)
        
        return threshs, tpr_fprs, roc_aucs, tpr_fprs_A , roc_aucs_A, len(text1s)
    
def plot_curves(input_paths, legend_labels, output_dir='./outputs'):
    os.makedirs(output_dir, exist_ok=True)
    all_data = {
        'tpr_fprs': [],
        'roc_aucs': [],
        'tpr_fprs_A': [],
        'roc_aucs_A': [],
        'threshs': None
    }
    input_list = []
    for input_path in input_paths:
        threshs, tpr_fprs, roc_aucs, tpr_fprs_A, roc_aucs_A, length = process_input(input_path)
        if threshs is not None:
            input_list.append(os.path.splitext(os.path.basename(input_path))[0])
            all_data['threshs'] = threshs
            all_data['tpr_fprs'].append(tpr_fprs)
            all_data['roc_aucs'].append(roc_aucs)
            all_data['tpr_fprs_A'].append(tpr_fprs_A)
            all_data['roc_aucs_A'].append(roc_aucs_A)
    metrics = ['tpr_fprs_A', 'roc_aucs_A', 'tpr_fprs', 'roc_aucs']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for metric in metrics:
        plt.figure(figsize=(6.5, 4))
        plt.xticks(all_data['threshs'], fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, which='both', linestyle=':', alpha=0.7)
        for idx, values in enumerate(all_data[metric]):
            style = linestyles[idx % len(linestyles)]
            marker = markers[idx % len(markers)]
            plt.plot(
                all_data['threshs'], values,
                linestyle=style,
                marker=marker,
                markersize=6,
                linewidth=1.2
            )
        plt.xlabel('Threshold', fontsize=10)
        if metric in ['tpr_fprs', 'tpr_fprs_A']:
            plt.ylabel('TPR at 1% FPR', fontsize=10)
        else:
            plt.ylabel('AUC', fontsize=10)
        plt.legend(legend_labels, loc='lower right', fontsize=10, frameon=False)
        plt.tight_layout()
        plt.savefig(
            f'{output_dir}/{metric}_{timestamp}.png',
            dpi=400,
            format='png',
            bbox_inches='tight'
        )
        plt.close()

if __name__ == '__main__':
    input_paths = [path.strip() for path in args.input_path.split(',')]

    legend_labels = ['POSTMARK@12(no iter.)', 'POSTMARK@12', 'ISOMARK@12(no iter.)', 'ISOMARK@12', 'mixed@12']  ###

    plot_curves(input_paths, legend_labels)