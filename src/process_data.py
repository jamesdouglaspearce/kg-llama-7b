import os
import glob
import json
import tqdm

if __name__ == '__main__':
    path = '../data'
    files = glob.glob('../data/*.tsv')
    print(f'files to process: {files}')
    for fname in files:
        text_output = []

        with open(fname, 'r') as f:
            for line in tqdm.tqdm(f.readlines()):
                d = json.loads(line)
                text_output.append(d['sentence'] + ' -> ' + json.dumps(d['triples']))
            print(f'Processed {fname}: {len(text_output)} examples..')

        fname_out = 'processed-' + fname.split('-')[-1].split('.')[0] + '.txt'
        print(f'Saving to {os.path.join(path, fname_out)}..')
        with open(os.path.join(path, fname_out), 'w') as f:
            f.write('\n'.join(text_output))

    print('...done')

