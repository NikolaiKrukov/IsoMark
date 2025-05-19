import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from embed import Paragram
from utils import token_pos
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import mannwhitneyu

paragram = Paragram(ratio=0.1)

def evaluate_replace(list1s, _text2s, _text3s, file_path='./data/count_watermark_word.json'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'r') as f:
            count_dict = json.load(f)
            count_dict = defaultdict(lambda: [0, 0, 0.0], count_dict)
    except:
        count_dict = defaultdict(lambda: [0, 0, 0.0])
    length = len(_text2s)
    for i in range(length):
        words = list1s[i]
        text2 = [token.text for token in _text2s[i]]
        text3 = [token.text for token in _text3s[i]]
        for word in words:
            if word in text2:
                if word not in text3:
                    count_dict[word][0] += 1
                count_dict[word][1] += 1
    for word in count_dict:
        count_dict[word][2] = count_dict[word][0] / count_dict[word][1]
    count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1][2]))  ### 修改根据哪个键排序
    with open(file_path, 'w') as f:
        json.dump(count_dict, f)

def sliding_std(scores, window_size=3):
    return np.max([np.std(scores[i:i+window_size]) for i in range(len(scores)-window_size+1)])

def evaluate(attack, _text1s, _text2s, _text3s, list1s, list2s, list3s, thresh, target_fpr=0.01, detect_type='normal'):
    score1s, score2s, score3s = [], [], []
    length = len(_text1s)
    for i in range(length):
        score1s.append(compute_presence(_text1s[i], list1s[i], threshold=thresh, detect_type=detect_type))
        score2s.append(compute_presence(_text2s[i], list2s[i], threshold=thresh, detect_type=detect_type))
        if attack:
            score3s.append(compute_presence(_text3s[i], list3s[i], threshold=thresh, detect_type=detect_type))

    labels = [0] * length + [1] * length

    scores = score1s + score2s
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_fpr = get_tpr_target(fpr, tpr, target_fpr=target_fpr)
    roc_auc = auc(fpr, tpr)
    print(f'--------------------------------------\nResult with threshold {thresh:5.3f}:')
    print(f'\tTPR at {target_fpr*100:3.1f}% FPR: {tpr_fpr:5.3f}%')
    print(f'\tAUC: {roc_auc:5.3f}')

    if attack:
        fp_rate = sum(np.array(score3s) >= thresh) / len(score3s)  # 若text3均为阴性（此时这个值和下面的值都无关，计算best没有意义）

        scores = score1s + score3s
        fpr_, tpr_, _ = roc_curve(labels, scores)
        tpr_fpr_ = get_tpr_target(fpr_, tpr_, target_fpr=target_fpr)
        roc_auc_ = auc(fpr_, tpr_)
        _, p_value = mannwhitneyu(score2s, score3s, alternative='greater')
        max_sliding_std = sliding_std(scores)
        print('attack:')
        print(f'\tTPR at {target_fpr*100:3.1f}% FPR: {tpr_fpr_:5.3f}%')
        print(f'\tAUC: {roc_auc_:5.3f}')
        #print(f'\tp-value: {p_value:5.3f}')
        # print(f'\tfp_rate (if text3s are all negative): {fp_rate*100:5.3f}%')
        print(f'\tsliding_std: {max_sliding_std:5.3f}')
        return tpr_fpr, roc_auc, tpr_fpr_, roc_auc_, score1s, score2s, score3s, p_value, fp_rate, max_sliding_std
    else:
        return tpr_fpr, roc_auc, score1s, score2s

def get_similarities(list_words, text_words):
    list_words_embs = paragram.get_embeddings(list_words)
    text_words_embs = paragram.get_embeddings(text_words)
    sims = []
    for list_word_emb in list_words_embs:
        sim = F.cosine_similarity(text_words_embs, list_word_emb.unsqueeze(0))
        if sim.shape[0] != len(text_words):
            assert text_words_embs.shape[0] != len(text_words), f'{sim.shape[0]} != {len(text_words)}'
        sims.append(sim)
    sims = torch.stack(sims, dim=0).cpu()
    topk_scores, topk_indices = torch.topk(sims, 1, dim=1)
    # topk_words = [[text_words[i] for i in indices] for indices in topk_indices]
    return topk_scores

def compute_presence(_text, words, threshold=0.7, detect_type='normal'):  # 依据全局嵌入选出list，计算words_in_text和list(词嵌入对)的相关性
    match detect_type:
        case 'normal' | 'normal_rotate':
            text_words = [token.text.lower() for token in _text if not token.is_punct and not token.is_space]
        case 'limit' | 'limit_rotate':
            text_words = [token.text.lower() for token in _text if not token.is_punct and not token.is_space and token_pos(token)]
        case _:
            raise NotImplementedError('Error detect_type.')
    topk_scores = get_similarities(words, text_words)
    present = 0.0
    for w, s in zip(words, topk_scores):
        if w.lower() in text_words or s >= threshold:  # 计算词嵌入对的相关性；保留等于号，这样就可以判断“完全相同”情况下的正确率了
            present += 1
    match detect_type:
        case 'normal' | 'limit':
            presence = present / len(words)
        case 'normal_rotate' | 'limit_rotate':
            presence = present / len(text_words)
        case _:
            raise NotImplementedError('Error detect_type.')
    return presence

def get_tpr_target(fpr, tpr, target_fpr=0.01):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    if target_fpr < fpr[0]:
        print(f'TPR at {target_fpr*100}% FPR: {tpr[0] * 100:5.1f}% (target too low)')
        return tpr[0] * 100
    
    if target_fpr > fpr[-1]:
        print(f'TPR at {target_fpr*100}% FPR: {tpr[-1] * 100:5.1f}% (target too high)')
        return tpr[-1] * 100
    
    idx = np.searchsorted(fpr, target_fpr, side='right')
    
    if fpr[idx-1] == target_fpr:
        tpr_value = tpr[idx-1]
    else:
        tpr_value = tpr[idx-1] + (target_fpr - fpr[idx-1]) * (tpr[idx] - tpr[idx-1]) / (fpr[idx] - fpr[idx-1])
    return tpr_value * 100