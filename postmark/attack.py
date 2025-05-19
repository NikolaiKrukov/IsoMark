import json
import random
from torch.nn.functional import cosine_similarity
from utils import load_template
from embed import nlp, Embed
from models import LLM

paraphrase_template = load_template('paraphrase_sent')
paraphrase_total_template = load_template('paraphrase_total')

class WatermarkAttacker:
    def __init__(self, attacker, freq=1000):
        self.attacker = LLM(attacker)
        # with open(f'./data/f{freq}/synonym_map-f{freq}.json', 'r') as f:
        #     self.synonym_map = json.load(f)
        # self.embedder = Embed(word=True)
    
    def paraphrase(self, text, prefix, max_tokens, temperature):
        doc = nlp(text)
        sentences = [str(s) for s in doc.sents]
        pp_sentences = [' ']
        for sentence in sentences:
            prompt = paraphrase_template.format(' '.join(pp_sentences), sentence)
            pp_sentence = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            pp_sentences.append(pp_sentence)
        pp_text = ' '.join(pp_sentences)
        return pp_text

    def paraphrase_paragraph(self, text, prefix, max_tokens, temperature, sentences_per_paragraph=3):
        doc = nlp(text)
        sentences = [str(s).strip() for s in doc.sents]
        n = len(sentences)
        pp_paragraphs = [' ']

        i = 0
        while i < n: # 当剩余句数 < 基准句数时，也要保证能至少分成一个段落
            group_size = min(max(1, sentences_per_paragraph + random.randint(-1, 1)), n - i)
            current_sentences = sentences[i : i + group_size]
            paragraph_text = ' '.join(current_sentences)

            prompt = paraphrase_template.format(' '.join(pp_paragraphs), paragraph_text)
            pp_paragraph = self.generate(prompt, max_tokens=3000, temperature=temperature)
            pp_paragraphs.append(pp_paragraph)
            
            i += group_size

        pp_text = ' '.join(pp_paragraphs)
        return pp_text
        
    def random_remove(self, text, prefix, max_tokens, temperature, ratio=0.2):
        doc = nlp(text)
        sentences = [str(s) for s in doc.sents]
        pp_sentences = [' ']
        for sentence in sentences:
            words = [token.text for token in nlp(sentence)]
            num_to_remove = int(len(words) * ratio)
            remove_indices = set(random.sample(range(len(words)), num_to_remove))
            ret_words = [word for i, word in enumerate(words) if i not in remove_indices]
            sentence = ' '.join(ret_words)

            # context = ' '.join(sentences[:i])
            prompt = paraphrase_template.format(' '.join(pp_sentences), sentence)
            pp_sentence = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            pp_sentences.append(pp_sentence)
        pp_text = ' '.join(pp_sentences)
        return pp_text
    
    # def replace_with_synonym(self, word, prefix, current_text, k=5):
    #     synonyms = self.synonym_map.get(word, [])[:k]
    #     if not synonyms:
    #         return [(word, 0.0)]
        
    #     synonym_texts = []
    #     for syn, _ in synonyms:
    #         modified_text = current_text.replace(word, syn)
    #         synonym_texts.append(modified_text)
        
    #     batch_embeddings = self.embedder.get_doc_embedding([current_text] + synonym_texts)
    #     original_embedding = batch_embeddings[0]
    #     synonym_embeddings = batch_embeddings[1:]
        
    #     sims = cosine_similarity(original_embedding.unsqueeze(0), synonym_embeddings, dim=1)
    #     return list(zip([syn[0] for syn in synonyms], sims.tolist()))

    # def semantic_replace(self, text, prefix, max_tokens, temperature, ratio=0.7):
    #     words = text.split()
    #     replace_count = int(len(words) * ratio)
        
    #     beams = [{'modified': words.copy(), 'score': 0.0, 'replaced': 0}]
        
    #     for _ in range(replace_count):
    #         new_beams = []
    #         for beam in beams:
    #             if beam['replaced'] >= replace_count:
    #                 new_beams.append(beam)
    #                 continue
                
    #             for i, word in enumerate(beam['modified']):
    #                 if word not in self.synonym_map:
    #                     continue
                    
    #                 candidates = self.replace_with_synonym(word, prefix, ' '.join(beam['modified'][i:]))
    #                 for syn, dist in candidates[:5]:
    #                     new_modified = beam['modified'].copy()
    #                     new_modified[i] = syn
    #                     new_beams.append({
    #                         'modified': new_modified,
    #                         'score': beam['score'] + dist,
    #                         'replaced': beam['replaced'] + 1
    #                     })

    #         beams = sorted(new_beams, key=lambda x: -x['score'])[:5]
        
    #     best_beam = beams[0]
    #     modified_text = ' '.join(best_beam['modified'])
    #     prompt = paraphrase_template.format(prefix, modified_text)
    #     return self.generate(prompt, max_tokens, temperature)
    
    def generate(self, prompt, max_tokens, temperature):
        return self.attacker.generate(prompt, max_tokens=max_tokens, temperature=temperature)

class Attack():
    def __init__(self, model='deepseek-v3', freq=1000):
        self.attacker = WatermarkAttacker(attacker=model, freq=freq)

    def generate_attack(self, attack, text, prefix, max_tokens=1000, temperature=0.0):
        match attack:
            case 'paraphrase':
                return self.attacker.paraphrase(text, prefix, max_tokens=max_tokens, temperature=temperature)
            case 'paragraph':
                return self.attacker.paraphrase_paragraph(text, prefix, max_tokens=max_tokens, temperature=temperature)
            case 'random_remove':
                return self.attacker.random_remove(text, prefix, ratio=0.2, max_tokens=max_tokens, temperature=temperature)
            case _:
                raise NotImplementedError('Error attack method.')