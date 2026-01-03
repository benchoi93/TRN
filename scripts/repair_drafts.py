import sqlite3
import json
import os

DB = 'data/survey.db'

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

rows = cur.execute('SELECT token, payload FROM drafts').fetchall()
if not rows:
    print('No drafts found.')
else:
    for row in rows:
        token = row['token']
        payload = json.loads(row['payload']) if row['payload'] else {}
        survey = cur.execute('SELECT json_path FROM surveys WHERE token=?', (token,)).fetchone()
        if not survey:
            print(f'No survey entry for token {token}, skipping')
            continue
        json_path = survey['json_path']
        if not os.path.exists(json_path):
            print(f'Paper JSON {json_path} not found for token {token}, skipping')
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            paper = json.load(f)
        candidates = paper.get('inspiring_references', []) or []
        # build search index on raw_match/title/first_author
        index = []
        for c in candidates:
            idx = c.get('id') or ''
            raw = (c.get('raw_match') or '')
            title = c.get('title') or ''
            fa = c.get('first_author') or ''
            index.append({'id': idx, 'raw': raw.lower(), 'title': title.lower(), 'fa': fa.lower()})

        changed = False
        answers = payload.get('answers', {})
        for page_key, ans in list(answers.items()):
            sels = ans.get('selected_ids', []) or []
            new_sels = []
            for s in sels:
                s_norm = (s or '').strip()
                if not s_norm:
                    continue
                # if it's already an id in candidates, keep
                found = None
                for entry in index:
                    if s_norm == entry['id']:
                        found = entry['id']
                        break
                if found:
                    new_sels.append(found)
                    continue
                # try substring match in raw
                s_l = s_norm.lower()
                for entry in index:
                    if s_l in entry['raw'] or s_l in entry['title'] or s_l in entry['fa']:
                        found = entry['id']
                        break
                if found:
                    new_sels.append(found)
                    continue
                # try reverse: raw contains snippet prefix (take first 30 chars)
                for entry in index:
                    if entry['raw'].startswith(s_l) or entry['title'].startswith(s_l):
                        found = entry['id']
                        break
                if found:
                    new_sels.append(found)
                    continue
                # no match -> skip
                print(f"Could not match selection '{s_norm}' for token {token} page {page_key}")
            # dedupe and keep order
            dedup = []
            for x in new_sels:
                if x not in dedup:
                    dedup.append(x)
            if dedup != sels:
                answers[page_key]['selected_ids'] = dedup
                changed = True
        if changed:
            payload['answers'] = answers
            cur.execute('INSERT INTO drafts(token,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at',
                        (token, json.dumps(payload, ensure_ascii=False),))
            conn.commit()
            print(f'Updated draft for token {token}')
        else:
            print(f'No changes for token {token}')

conn.close()
