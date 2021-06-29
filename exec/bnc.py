import json
import os

from pyconversations.reader import BNCReader

if __name__ == '__main__':
    data_root = '/Users/hsh28/data/'
    out = data_root + 'conversations/'

    os.makedirs(out + 'Reddit/BNC/', exist_ok=True)
    convos = BNCReader.read(data_root + 'BNC/*', ld=True)

    cache = []
    messages = {
        'all': 0,
        'ah': 0,
        'non': 0
    }
    conversations = {
        'all': 0,
        'ah': 0,
        'non': 0,
    }
    for convo in convos:
        cache.append(json.dumps(convo.to_json()))

        ah = False
        for post in convo.posts.values():
            messages['all'] += 1
            if 'AH=1' in post.tags:
                messages['ah'] += 1
                ah = True
            else:
                messages['non'] += 1

        conversations['all'] += 1
        if ah:
            conversations['ah'] += 1
        else:
            conversations['non'] += 1

    with open(out + 'Reddit/BNC/out.json', 'w+') as fp:
        fp.write('\n'.join(cache))

    print(messages)
    print(conversations)
