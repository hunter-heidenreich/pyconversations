"""
This file represents an executable that demonstrates
conversion from a raw representation
into a conversation segmented, JSON-line separated output
for further downstream analysis.
"""
import json
import os

from pyconversations.reader import RawFBReader

if __name__ == '__main__':
    data_root = '/Users/hsh28/data/'
    out = data_root + 'conversations/'

    # Pre-process BuzzFace
    name = 'buzzface'
    for pagename, convo_chunk in RawFBReader.iter_read(data_root + 'BuzzFace/data*/'):
        print(f'{pagename}: {len(convo_chunk)} conversations')
        lines = [json.dumps(convo.to_json()) for convo in convo_chunk]

        os.makedirs(out + 'BuzzFace', exist_ok=True)
        with open(out + f'BuzzFace/{pagename}.json', 'w+') as fp:
            fp.write('\n'.join(lines))
