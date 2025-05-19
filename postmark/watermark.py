# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import re
import argparse
import threading
import traceback
from tqdm import tqdm
import torch
import json
import random
import nltk
from models import LLM
from embed import Embed
from utils import plot_distribution, dataset_get, load_template
from attack import Attack

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='Which dataset to use.', choices=['opengen', 'factscore', 'lfqa'], default='opengen')
parser.add_argument('--output_path', type=str, help='JSONL path to save output data.', default=None)

parser.add_argument('--model', type=str, help='The base LLM to watermark the outputs of.', choices=['gpt-4', 'llama-3-8b', 'llama-3-8b-chat', 'mistral-7b-inst', 'deepseek-r1-distill-llama-8b'], default='deepseek-r1-distill-llama-8b')
parser.add_argument('--embedder', type=str, help='The embedding model for selecting watermark words.', choices=['openaiemb', 'nomic', 'qwen'], default='nomic')
parser.add_argument('--function', type=str, help='Normal: POSTMARK\'s function type.', choices=['normal', 'random'], default='normal')
parser.add_argument('--embed_type', type=str, help='Origin: Postmark\'s embedding word list. Isolated: Isolated embedding word list (Improvement).', choices=['origin', 'isolated', 'mixed', 'filtered'], default='origin')
parser.add_argument('--inserter', type=str, help='The LLM for inserting watermark words.', choices=['gpt-4o', 'deepseek-v3', 'qwen-max', 'qwen-max-latest', 'qwen-max-2024-09-19'], default='deepseek-v3')

parser.add_argument('--ratio', type=float, help='Determines how many watermark words to insert.', default=0.12)
parser.add_argument('--freq', type=int, help='Determines the lowerbound of words\'s frequency.', default=1000)
parser.add_argument('--up', type=float, help='Determines the upperbound of isolated words\'s top-1 similarity.', default=1.0)
parser.add_argument('--down', type=float, help='Determines the lowerbound of isolated words\'s top-1 similarity.', default=0.6)
parser.add_argument('--alpha', type=float, help='Determines the proportion of origin watermark words in mixed method.', default=0.3)
parser.add_argument('--freq_thresh', type=float, help='Determines the proportion of origin watermark words in filtered method.', default=1.0)

parser.add_argument('--iterate', type=int, help='The step of the iterative insertion. 0 means no iteration.', default=0)
parser.add_argument('--iterate_type', type=str, help='v1: Use normal iterate method. v2: Attempt to obtain greater sub_presence using origin iteration method.', choices=['v1', 'v2', 'v0'], default='v1')

parser.add_argument('--attack', type=str, help='Which attack method.', choices=['none', 'random_remove', 'paraphrase', 'paragraph'], default='none')
parser.add_argument('--attacker', type=str, help='Which attack model.', choices=['gpt-3.5-turbo', 'gpt-4o', 'qwen-max', 'qwen-turbo', 'deepseek-v3'], default='deepseek-v3')

parser.add_argument('--m', type=int, help='Number of thread to use.', default=1)
parser.add_argument('--s', type=int, help='LHS of instances to input.', default=0)
parser.add_argument('--n', type=int, help='RHS of instances to input.', default=0)

parser.add_argument('--cache_text1', type=str, help='(Optional) Path to a JSONL file that contains existing text1s to be reused', default=None)
parser.add_argument('--cache_text2', type=str, help='(Optional) Path to a JSONL file that contains existing text2s to be reused', default=None)

args = parser.parse_args()
print(args)

if args.cache_text1 and args.cache_text2:
    raise ValueError('Do not provide cache_text1 and cache_text2 at the same time.')

if args.n and args.s >= args.n:
    raise ValueError('LHS of instances cannot be greater than or equal to RHS of instances.')

if args.output_path:
    output_path = args.output_path
else:
    output_path = f'./outputs/test_{args.embedder}_r{args.ratio}_f{args.freq}'
    if args.function == 'normal':
        output_path += f'_{args.embed_type}'
        match args.embed_type:
            case 'origin':
                pass
            case 'isolated':
                output_path += f'_u{args.up}_d{args.down}'
            case 'mixed':
                output_path += f'_u{args.up}_d{args.down}_a{args.alpha}'
            case 'filtered':
                output_path += f'_ft{args.freq_thresh}'
    else:
        output_path += f'_{args.function}'
        args.embed_type = 'origin'
        
    if args.iterate:
        output_path += f'_i{args.iterate}-{args.iterate_type}'
    if args.attack:
        output_path += f'_{args.attack}-{args.attacker}'
    output_path += '.jsonl'
print(f'output_path: {output_path}')

file_lock1 = threading.Lock()
file_lock2 = threading.Lock()
file_lock3 = threading.Lock()

Presence11 = []
Presence12 = []
Presence22 = []

prefix = 'title' if args.dataset == 'factscore' else 'prefix'

def append_to_output_file(generation_records):
    with file_lock3:
        with open(output_path, 'a') as f:
            for record in generation_records:
                f.write(json.dumps(record) + '\n')
            f.flush()

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', text.lower())

class Watermarker():
    def __init__(self):
        self.model = LLM(args.model)
        self.embedder = Embed(model=args.embedder, ratio=args.ratio, freq=args.freq, up_thresh=args.up, down_thresh=args.down, embed_type=args.embed_type, alpha=args.alpha, freq_thresh=args.freq_thresh)  # 千万不要混用不同词库下的Embed!!!idx2vec会存在重大问题!!!（就算都是isolated且只差10个单词也会有影响）
        self.inserter = LLM(args.inserter)
        self.function = args.function
        self.iterate = args.iterate
        self.iterate_type = args.iterate_type
        self.template = load_template('insert')
        if args.attack != 'none':
            self.attacker = Attack(model=args.attacker, freq=args.freq)

    def get_words(self, text):
        words = self.embedder.get_words(text, function=self.function)
        return words
    
    def insert_watermark(self, text1, max_tokens=1500, sub_presence_thresh = 0.5, max_attempt = 3):
        if not text1.strip():
            raise ValueError('Text1 is empty.')
        
        list1 = self.get_words(text1)
        clean_text1 = clean_text(text1)  # clean_text和get_words无关，只用于计算presence11
        presence11 = sum([1 for word in list1 if word in clean_text1]) / len(list1)
        # print(f'{presence11 * 100:5.3f}% of list1 are in text1')
        if list1 is None or len(list1) == 0:
            print('No words found in list1, returning...')
            return {'text1': text1, 'list1': list1, 'text2': text1, 'list2': list1}, presence11, [], []  # 所有输出都是这种结构的话，TPR at 1.0% FPR: 1.000% & AUC: 0.500，即使list1不为空

        if self.iterate:
            sublists = [list1[i:i+self.iterate] for i in range(0, len(list1), self.iterate)]  # python中切片会自动取min，不会引发错误
            input_res = text1
            if self.iterate_type == 'v2':  # 由于v1测试中尚未出现presence12 < 0.65的情况，并且 < 0.80的情况只有一次，并且大部分presence12都在0.85以上，故v2近似于v1
                for sublist in sublists:
                    new_prompt = self.template.format(input_res, ', '.join(sublist))
                    sub_presence = 0
                    n_attempts = 0
                    while sub_presence < sub_presence_thresh:
                        if n_attempts >= max_attempt:
                            print(f'Exceeded {max_attempt} tries, breaking...sub_presence: {sub_presence}')
                            break
                        input_res = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
                        clean_input_res = clean_text(input_res)
                        sub_presence = sum([1 for word in sublist if word in clean_input_res]) / len(sublist)
                        n_attempts += 1
            elif self.iterate_type == 'v1':
                for sublist in sublists:
                    new_prompt = self.template.format(input_res, ', '.join(sublist))
                    input_res = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
            elif self.iterate_type == 'v0':
                for sublist in sublists:
                    new_prompt = self.template.format(input_res, ', '.join(sublist))
                    input_res = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
            else:
                raise NotImplementedError('Error iterate_type.')

            text2 = input_res
        else:
            new_prompt = self.template.format(text1, ', '.join(list1))
            text2 = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)

        if text2 is None or len(text2) == 0:
            print('No words found in text2, returning...')
            return {'text1': text1, 'list1': list1, 'text2': text1, 'list2': list1}, presence11, [], []
        list2 = self.get_words(text2)
        if list2 is None or len(list2) == 0:
            print('No words found in list2, returning...')
            return {'text1': text1, 'list1': list1, 'text2': text1, 'list2': list1}, presence11, [], []

        clean_text2 = clean_text(text2)
        presence12 = sum([1 for word in list1 if word in clean_text2]) / len(list1)
        # print(f'{presence12 * 100:5.3f}% of list1 are in text2')
        presence22 = sum([1 for word in list2 if word in clean_text2]) / len(list2)
        # print(f'{presence22 * 100:5.3f}% of list2 are in text2')

        res = {
            'text1': text1,
            'list1': list1,
            'text2': text2,
            'list2': list2
        }
        return res, presence11, presence12, presence22

def process_text(thread_id, chunks, template, paragraph_mode=True, sentences_per_paragraph=4):
    try:
        with file_lock1:
            watermarker = Watermarker()

        generation_records = []
        for idx, dd in tqdm(enumerate(chunks)):
            if args.cache_text2:
                record = dd
            else:
                if args.cache_text1:
                    text1 = dd['text1']  # 不能使用不同embedder的list1，必须重新生成
                else:
                    prompt = template.format(dd['prefix'])  # 作者的prefix并没有包含template
                    text1 = watermarker.model.generate(prompt, max_tokens=1500, temperature=1)

                if not paragraph_mode:
                    # ---------- 全文级插入 ----------
                    record, presence11, presence12, presence22 = (watermarker.insert_watermark(text1, max_tokens=1500))
                else:
                    # ---------- 段落级插入 ----------
                    sentences = nltk.tokenize.sent_tokenize(text1)
                    paragraphs = []
                    for i in range(0, len(sentences), sentences_per_paragraph):
                        chunk_sents = sentences[i:i + sentences_per_paragraph]
                        paragraphs.append(" ".join(chunk_sents))
                    merged_text2_parts = []
                    merged_list2 = []
                    total_presence11 = 0
                    total_presence12 = 0
                    total_presence22 = 0

                    for para in paragraphs:
                        part_record, p11, p12, p22 = (
                            watermarker.insert_watermark(para, max_tokens=1500)
                        )
                        merged_text2_parts.append(part_record.get('text2', ''))
                        merged_list2.extend(part_record.get('list2', []))
                        total_presence11 += p11
                        total_presence12 += p12
                        total_presence22 += p22

                    combined_text2 = "\n".join(merged_text2_parts)
                    record = {
                        'text1': text1,
                        'text2': combined_text2,
                        'list2': merged_list2,
                    }
                    presence11 = total_presence11
                    presence12 = total_presence12
                    presence22 = total_presence22
                
                dd.update(record)  # update是原地操作
                record = dd

                with file_lock2:
                    Presence11.append(presence11)
                    Presence12.append(presence12)
                    Presence22.append(presence22)

            if args.attack != 'none':
                text3 = watermarker.attacker.generate_attack(args.attack, record['text2'], record['prefix'])
                list3 = watermarker.get_words(text3)
                record.update({'text3': text3, 'list3': list3})
            generation_records.append(record)

            if len(generation_records) >= 50:
                append_to_output_file(generation_records)
                generation_records = []
        
        if generation_records:
            append_to_output_file(generation_records)

        print(f'Thread {thread_id} finished processing.')
    except Exception as e:  # 线程的错误都是静默失败的，所以需要try-except + traceback.print_exc()来给出完整错误信息
        print(f'Thread {thread_id} encountered an error: {e}')
        traceback.print_exc()

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    template = None
    if args.cache_text2:
        with open(args.cache_text2, 'r') as f:
            records = []
            for i, line in enumerate(f):  # 逐行读取，不一次性加载全部内容
                if i < args.s:
                    continue
                if args.n and i >= args.n:
                    break
                
                data = json.loads(line)
                records.append({
                    'prefix': data[prefix],
                    'text1': data['text1'],
                    'list1': data['list1'],
                    'text2': data['text2'],
                    'list2': data['list2']
                })
        print(f'Loaded {len(records)} text1s, list1, text2s and list2s examples for {args.cache_text2}.')
    elif args.cache_text1:
        with open(args.cache_text1, 'r') as f:
            records = []
            for i, line in enumerate(f):
                if i < args.s:
                    continue
                if args.n and i >= args.n:
                    break
                data = json.loads(line)
                records.append({
                    'prefix': data[prefix],
                    'text1': data['text1'],
                    'list1': data['list1']
                })
        print(f'Loaded {len(records)} text1s examples for {args.cache_text1}.')
    else:
        input_path, template = dataset_get(args.dataset)
        
        with open(input_path, 'r') as f:
            records = []
            for i, line in enumerate(f):
                if i < args.s:
                    continue
                if args.n and i >= args.n:
                    break
                records.append({'prefix': json.loads(line)[prefix]})
        print(f'Loaded {len(records)} data examples from {args.dataset}.')

    if args.m <= 1:
        process_text(0, records, template)
    else:
        chunks = [records[i::args.m] for i in range(args.m)]
        
        threads = []
        for i in range(args.m):
            thread = threading.Thread(target=process_text, args=(i, chunks[i], template))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    print('All threads have finished processing.')
    print(f'Output the results to {output_path}.')

    if not args.cache_text2:
        plot_distribution(Presence11, 'presence_list1_text1', './outputs/presence/')
        plot_distribution(Presence12, 'presence_list1_text2', './outputs/presence/')
        plot_distribution(Presence22, 'presence_list2_text2', './outputs/presence/')