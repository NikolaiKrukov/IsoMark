import os
import numpy as np
import nltk
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
tagger = nltk.PerceptronTagger()

def word_pos(word):  # nltk
    tag = tagger.tag([word])[0][1]  # 添加一个[]为妙
    return 'NNP' not in tag and tag in ['NN', 'VB', 'JJ', 'RB']  # 通过删除所有非词元词汇的方式，保证所有词汇满足word==lemma

def token_pos(token):  # spacy
    return (
        token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}
        and token.tag_ not in {'IN', 'TO', 'DT', 'PRP', 'PRP$', 'MD'}
    )

def load_template(file_name):
    with open(f'./prompts/{file_name}.txt') as f:
        template = f.read().strip()
    return template

def plot_distribution(data, name, output_path):
    mu, std = np.mean(data), np.std(data)
    
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Distribution\n(μ={mu:.1f}, σ={std:.1f})')
    
    plt.title(f'Distribution of {name}', fontsize=16)
    plt.xlabel('Value (%)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = os.path.join(output_path, f'{name}_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to: {filename}')

def dataset_get(dataset):
    if dataset == 'opengen':
        input_path = './data/opengen.jsonl'
        template = load_template('opengen')
    elif dataset == 'lfqa':
        input_path = './data/lfqa.jsonl'
        template = load_template('lfqa')
    elif dataset == 'factscore':
        input_path = './data/factscore.jsonl'
        template = load_template('factscore')
    else:
        raise ValueError
    return input_path, template