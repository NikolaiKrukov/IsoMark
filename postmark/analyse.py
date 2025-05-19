import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import re
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from collections import Counter
from utils import load_template
from embed import nlp, Embed
from models import LLM

quality_template = load_template('quality')
embedder = Embed(word=True)

def process_text(text):
    tokens = nlp(text)
    words = [token.text for token in tokens]
    embeddings = embedder.get_doc_embedding(words).cpu()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, words, tokens

def compare_similarity(text1, text2):
    emb1 = embedder.get_doc_embedding([text1])
    emb2 = embedder.get_doc_embedding([text2])
    return cosine_similarity(emb1, emb2, dim=1)

def evaluate_score(input_path, quality=True, similarity=True, type='12'):
    if quality:
        a_cnt = 0
        b_cnt = 0
        c_cnt = 0
    if similarity:
        sims = []

    with open(input_path, 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            if idx >= 100:
                break
            
            data = json.loads(line)

            if quality:
                comparison_quality = compare_text_quality(data['prefix'], data[f'text{type[0]}'], data[f'text{type[1]}'])  # 这里的prefix和watermark.py里一样不包含template，但是影响应该不大
                match = re.findall(r'\[\[(.*?)\]\]', comparison_quality)[0]
                if match:
                    if match == 'A':
                        a_cnt += 1
                    elif match == 'B':
                        b_cnt += 1
                    else:
                        c_cnt += 1

            if similarity:
                comparison_similarity = compare_similarity(data[f'text{type[0]}'], data[f'text{type[1]}']).cpu().numpy()
                sims.append(comparison_similarity)

    if quality:
        print(a_cnt, b_cnt, c_cnt)
        pct = (b_cnt + c_cnt*0.5) / (a_cnt + b_cnt + c_cnt) * 100
        print(f'\n\tSoft win rate: {pct:.2f}')

    if similarity:
        avg_sim = np.mean(sims) * 100
        print('\tavg_sim: ', avg_sim)

def compare_text_quality(prefix, text1, text2):
    model = LLM('deepseek-v3')
    prompt = quality_template.format(prefix, text1, text2)
    result = model.generate(prompt, max_tokens=3000, temperature=0.0)
    print(result)
    result = re.sub(r'^```(?:json)?|```$', '', result, flags=re.MULTILINE).strip()
    if result:
        return result
    else:
        return None

if __name__ == '__main__':
    evaluate_score('outputs/test_nomic_r0.06_f1000_isolated_u1.0_d0.6_i20-v1_paraphrase-deepseek-v3.jsonl')