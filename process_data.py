from datasets import load_dataset
from unidecode import unidecode

ds = load_dataset("huggingartists/justin-bieber")

with open('Bieber-bible.txt', 'w') as file:
    for train in ds['train']:
        file.write(unidecode(train['text']) + '\n')