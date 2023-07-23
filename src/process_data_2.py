import os
import glob
import json
import tqdm
import random

if __name__ == '__main__':
    fname = '../data/elm_generated_corpus.jsonl'
    text_output = []

    with open(fname, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            d = json.loads(line)
            if 'triples' in d and 'gen_sentence' in d:
                text_output.append(d['gen_sentence'] + ' -> ' + json.dumps(d['triples']))

    random.shuffle(text_output)
    print(f'Processed {fname}: {len(text_output)} examples..')

    fname_out = '../data/processed-train.txt'
    print(f'Saving to {fname_out}..')
    with open(fname_out, 'w') as f:
        f.write('\n'.join(text_output))

    print('...done')

