from tinydb import TinyDB, Query
from glob import glob
import json

# db = TinyDB('output/db.json')

db = TinyDB('output/db.json', indent=2)
db.truncate()

# exps = db.table('experiments')

files = glob('output/*model*.json')

for f in files:
    with open(f, 'r') as json_file:
        metadata = json.load(json_file)
    metadata['dataFile'] = f.split('/')[-1]
    db.insert(metadata)

def set_nested(path, val):
    def transform(doc):
        for key in path[:-1]:
            doc = doc[key]
        doc[path[-1]] = val

    return transform

db.close()
# db.truncate()