import os
import numpy as np
import torch
import torch.nn.functional as F
import json
import spacy
import random
from collections import defaultdict
from torch.nn.functional import cosine_similarity
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import pickle
from models import openai_key, qwen_key
from utils import word_pos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en_core_web_sm')  # spacy对大小写敏感
spacy.prefer_gpu()

torch.manual_seed(42)
random.seed(42)

class EmbeddingModel():
    def __init__(self, ratio=0.12, freq=1000, up_thresh=1.0, down_thresh=0.6, alpha=0.3, freq_thresh=1.0):
        self.ratio = ratio
        self.freq = freq
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.alpha = alpha
        self.freq_thresh = freq_thresh

        self.nlp = nlp
        self.word2idx = defaultdict(list)
        self.idx2word = defaultdict(list)
        self.embedding_table = defaultdict(list)
    
    def get_doc_embedding(self, text):
        raise NotImplementedError('get_doc_embedding')
    
    def get_embeddings_batched(self, words, batch_size=64):
        raise NotImplementedError('get_embeddings_batched')
    
    def idx_to_word(self, idx, name):
        return self.idx2word[name][idx]
    
    def word_to_idx(self, word, name):
        return self.word2idx[name][word]
    
    def random_project(self, words, name):
        with open('./data/filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl', 'rb') as f:
            redpj_embs = pickle.load(f)
        indices = random.sample(range(len(redpj_embs)), len(words))
        emb_list = [redpj_embs[i].clone().detach() for i in indices]
        random.shuffle(emb_list)
        emb_table = torch.stack(emb_list)
        self.embedding_table[name] = emb_table.to(device)
        print(f'Embedding table shape ({name}): {self.embedding_table[name].shape}')  # 第0维就是总词数
        self.word2idx[name] = {word: idx for idx, word in enumerate(words)}
        self.idx2word[name] = words

    def get_words(self, text, function='normal'):
        if text is None:
            return None
        k = max(1, int(len(text.split()) * self.ratio))

        match function:
            case 'normal':
                if 'other' in self.word2idx:
                    k_ = int(k * self.alpha)
                    return self.get_words_once(text, k_, 'isolated') + self.get_words_once(text, k-k_, 'other')
                elif 'origin' in self.word2idx:
                    return self.get_words_once(text, k, 'origin')
                elif 'isolated' in self.word2idx:
                    return self.get_words_once(text, k, 'isolated')
                else:
                    raise ValueError('self.word2idx is empty.')
            case 'random':
                return self.get_words_once(text, k, 'origin', random=True)
            case _:
                raise ValueError('Error function type.')
    
    def get_words_once(self, text, k, name, a=3, random=False):  # 原prompt可能会导致LLM将水印词添加上** **加粗符号，导致全局嵌入异常，但是仍然获得了较高的AUC（nomic判断筛除所有*符号前后的文本相似度能达到0.9971，所以问题不大）
        if random:
            random.seed()
            words = random.sample(self.idx2word[name], k*a)
        else:
            response_vec = self.get_doc_embedding(text)
            if isinstance(response_vec, list):
                response_vec = response_vec.clone().detach().to(device)

            scores = cosine_similarity(self.embedding_table[name], response_vec, dim=1)  # 计算全局嵌入与词嵌入的相关性
            assert scores.shape[0] == len(self.idx2word[name]), f'scores shape mismatch with idx2word size: {scores.shape[0]} != {len(self.idx2word)}'
            if len(scores) < k*a:
                top_k_indices = torch.arange(len(scores), dtype=torch.long)
                print(f'len(scores) < k * {a} !!!')
            else:
                top_k_scores, top_k_indices = torch.topk(scores, k*a)
            
            top_k_words = [self.idx_to_word(index.item(), name) for index in top_k_indices]
            words = top_k_words

        try:
            word_embs = self.get_doc_embedding(words)
        except:
            return words
        word_embs = word_embs.clone().detach().to(device)
        text_emb = response_vec.unsqueeze(0).to(device)
        scores = cosine_similarity(text_emb, word_embs, dim=1)

        if len(scores) < k:
            top_k_indices = torch.arange(len(scores), dtype=torch.long)
            print('len(scores) < k !!!')
        else:
            top_k_scores, top_k_indices = torch.topk(scores, k)

        # print(top_k_scores)  # 无论origin还是isolated，这些词汇和原文本的相似度不是很高，大部分是0.3x，基本都低于0.45

        words = [words[idx] for idx in top_k_indices]
        words = sorted(list(set(words)))
        return words
    
    def generate_data(self):
        path_dir = f'./data/f{self.freq}'
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        
        # prepare high_freq_words
        self.wpath = f'{path_dir}/valid_wtmk_words_in_wiki_base-only-f{self.freq}.pkl'
        if not os.path.exists(self.wpath):
            final_list = self.generate_high_freq_words()
            with open(self.wpath, 'wb') as f:
                pickle.dump(final_list, f)

        # prepare synonym_map
        self.syn_mpath = f'{path_dir}/synonym_map-f{self.freq}.json'
        if not os.path.exists(self.syn_mpath):
            synonym_map = self.generate_synonym_mapping()
            with open(self.syn_mpath, 'w') as f:
                json.dump(synonym_map, f)

        # prepare isolated_words
        self.iso_wpath = f'{path_dir}/isolated_words-f{self.freq}_u{self.up_thresh}_d{self.down_thresh}.pkl'
        self.other_wpath = f'{path_dir}/other_words-f{self.freq}_u{self.up_thresh}_d{self.down_thresh}.pkl'
        if not os.path.exists(self.iso_wpath) or not os.path.exists(self.other_wpath):
            isolated_list, other_list = self.generate_isolated_words()
            with open(self.iso_wpath, 'wb') as f:
                pickle.dump(isolated_list, f)
            with open(self.other_wpath, 'wb') as f:
                pickle.dump(other_list, f)
        
        # prepare filtered_words
        self.freq_wpath = f'./data/f{self.freq}/filtered_words_f{self.freq}_ft{self.freq_thresh}.pkl'
        if not os.path.exists(self.freq_wpath):
            filtered_list = self.generate_filtered_words()
            with open(self.freq_wpath, 'wb') as f:
                pickle.dump(filtered_list, f)
    
    def generate_high_freq_words(self):
        with open( './data/wikitext_freq.json', 'r') as f:
            freq_dict = json.load(f)
        freq_dict = {w: f for w, f in freq_dict.items() if len(w) >= 3 and w.isalpha() and w.islower() and f >= self.freq}  # 改变频率的测试暂时没有什么效果
        freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
        all_words = list(freq_dict.keys())
        final_list = [word for word in all_words if word_pos(word)]
        return final_list

    def generate_synonym_mapping(self, k=100, chunk_size=128):
        with open(self.wpath, 'rb') as f:
            words = pickle.load(f)
        word_embeddings = self.get_doc_embedding(words).clone().detach().to(device)

        vocab_size = len(words)
        k = min(k, vocab_size - 1)
        synonym_map = defaultdict(list)
        
        for i in range(vocab_size):
            similarities = []
            for j in range(0, vocab_size, chunk_size):
                chunk = word_embeddings[j:j+chunk_size]
                chunk_sim = cosine_similarity(
                    word_embeddings[i].unsqueeze(0).unsqueeze(0),  # [1, 1, hidden_dim]
                    chunk.unsqueeze(0),  # [1, chunk_size, hidden_dim]
                    dim=2
                ).squeeze(0).squeeze(0)  # [chunk_size]
                similarities.append(chunk_sim)
            
            sim_scores = torch.cat(similarities)  # [vocab_size]
            
            _, top_indices = torch.topk(sim_scores, k=k+1)
            ranked = [(words[j], float(sim_scores[j])) for j in top_indices if j != i]
            synonym_map[words[i]] = ranked
        
        return synonym_map
    
    def generate_isolated_words(self):
        with open(self.syn_mpath, 'r') as f:
            data = json.load(f)
        isolated_list = [word for word, sims in data.items() if not any(self.down_thresh < sim[1] < self.up_thresh for sim in sims)]  # 基于静态地根据语义断崖筛选词汇；设定为(0.6, 1.0)时，词库变为1/10（3k->0.3k）；设定为(0.5, 1.0)时，词库只有3个词
        other_list = [word for word in data.keys() if word not in isolated_list]
        return isolated_list, other_list

    def generate_filtered_words(self):
        with open('./data/count_watermark_word.json', 'r') as f:  ###
            data = json.load(f)
        with open(self.wpath, 'rb') as f:
            words = pickle.load(f)
        filtered_list = [word for word in words if word in data and data[word][2] >= self.freq_thresh]
        return filtered_list

class NomicEmbed(EmbeddingModel):
    def __init__(self, model='nomic-embed-text-v1', word=False, embed_type='origin', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', trust_remote_code=True)
        self.embedder = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True).to(device)
        self.embedder.eval()

        self.generate_data()

        if not word:
            match embed_type:
                case 'origin':
                    with open(self.wpath, 'rb') as f:
                        words = pickle.load(f)
                    self.random_project(words, 'origin')
                case 'isolated':
                    with open(self.iso_wpath, 'rb') as f:
                        words = pickle.load(f)
                    self.random_project(words, 'isolated')
                case 'mixed':
                    with open(self.iso_wpath, 'rb') as f:
                        iso_words = pickle.load(f)
                    with open(self.other_wpath, 'rb') as f:
                        other_words = pickle.load(f)
                    self.random_project(iso_words, 'isolated')
                    self.random_project(other_words, 'other')
                case 'filtered':
                    with open(self.freq_wpath, 'rb') as f:
                        filtered_words = pickle.load(f)
                        self.random_project(filtered_words, 'origin')  ###
                case _:
                    raise NotImplementedError('Error embed_type.')
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_doc_embedding(self, text):
        if type(text) == str:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = self.embedder(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)[0]
        elif type(text) == list:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = self.embedder(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def get_embeddings_batched(self, words, batch_size=64):
        all_embeddings = []
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            batch_embeddings = self.get_doc_embedding(batch)
            all_embeddings.append(batch_embeddings)
    
        return np.vstack(all_embeddings)

class Embed():
    def __init__(self, model='nomic', ratio=0.12, freq=1000, up_thresh=1.0, down_thresh=0.6, word=False, embed_type='origin', alpha=0.3, freq_thresh=1.0):
        print(f'Loading {model} embedder...')
        if model == 'nomic':
            self.embedder = NomicEmbed(ratio=ratio, freq=freq, up_thresh=up_thresh, down_thresh=down_thresh, word=word, embed_type=embed_type, alpha=alpha, freq_thresh=freq_thresh)
        # elif model == 'openaiemb':
        #     self.embedder = OpenAIEmb(ratio=ratio, word=word)
        # elif embedder == 'openai':
        #     self.embedder = 
        # elif 'qwen' or 'deepseek' in model:
        #     self.embedder = QwenEmbed(ratio=ratio, word=word)
        else:
            raise NotImplementedError(f'No model named {model}.')
        
    def get_words(self, text, function='normal'):
        return self.embedder.get_words(text, function=function)
    
    def get_doc_embedding(self, text):
        return self.embedder.get_doc_embedding(text)
    
    def get_embeddings_batched(self, words, batch_size=64):
        return self.embedder.get_embeddings_batched(words, batch_size=batch_size)

class Paragram(EmbeddingModel):
    def __init__(self, filter_vocab=0, **kwargs):
        super().__init__(**kwargs)
        with open('./data/paragram_xxl.pkl', 'rb') as f:
            We = pickle.load(f)
        with open('./data/paragram_xxl_words.json', 'r') as f:
            words = json.load(f)
        self.word2idx = words
        self.idx2word = list(self.word2idx.keys())
        self.embedding_table = We.to(device)
        indices = list(self.word2idx.values())
    
    def idx_to_word(self, idx):
        return self.idx2word[idx]
    
    def word_to_idx(self, word):
        return self.word2idx[word]
    
    def get_word(self, embedding):
        idx = (self.embedding_table == embedding).all(dim=1).nonzero(as_tuple=True)[0]
        return self.idx_to_word(idx)

    def get_embedding(self, word):
        if word in self.word2idx:
            return self.embedding_table[self.word2idx[word]]
        else:
            return None
    
    def get_embeddings(self, words):
        indices = []
        none_indices = []
        for i, word in enumerate(words):
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                none_indices.append(i)
        embeddings = self.embedding_table[indices, :]
        if none_indices:
            for i in none_indices:
                t = torch.zeros(1, self.embedding_table.shape[1]).to(device)
                embeddings = torch.cat((embeddings[:i], t, embeddings[i:]), dim=0)
        assert embeddings.shape[0] == len(words), f'embeddings shape mismatch with words size: {embeddings.shape[0]} != {len(words)}'
        return embeddings
    
    def get_doc_embedding(self, text):
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_stop and token.text in self.word2idx]
        embeddings = self.get_embeddings(words)
        return embeddings.mean(dim=0).unsqueeze(0).to(device)
    
    def generate_data(self, freq, k, up_thresh, down_thresh):
        raise NotImplementedError('You should not use generate_data function here.')