import csv
from collections import Counter

rows = list(csv.DictReader(open(r'D:\Repos\opencode-shit\data\corpus_lmstudio.csv', encoding='utf-8')))
print(f'Total: {len(rows)}')
print(f'Unique: {len(set(r["name"].lower() for r in rows))}')

c = Counter(r['category'] for r in rows)
for cat, n in c.most_common():
    print(f'  {cat}: {n}')

print('\nSamples by category:')
seen = set()
for r in rows:
    cat = r['category']
    if cat not in seen:
        seen.add(cat)
        print(f'\n[{cat}]')
        for x in [r2 for r2 in rows if r2['category']==cat][:5]:
            print(f'  - {x["name"]}')

lens = [len(r['name'].split()) for r in rows]
print(f'\nLength: min={min(lens)} avg={sum(lens)/len(lens):.2f} max={max(lens)}')
