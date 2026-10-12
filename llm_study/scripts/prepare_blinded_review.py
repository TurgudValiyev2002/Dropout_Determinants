"""Build a frozen, method-blinded human review pack. Standard library only.

No inference, outcome access, answer rewriting, or automatic rating.
Only share reviewer_files; private_DO_NOT_SHARE contains the method key.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

EXPECTED_PROTOCOL = 'b19e9f81939fc4ddf9cba902388ba02bdbea456e11b4ba70fb7c9137a46f20a0'
STAGES = ('A_pre_enrollment', 'B_first_semester', 'C_second_semester')
FLAGS = ('contributor_omission', 'value_omission', 'direction_omission',
         'numerical_error', 'unsupported_claim', 'causal_probability_confusion',
         'missing_safeguard')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def indexed(path):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    result = {r['case_id']: r for r in records}
    require(len(result) == len(records), 'Duplicate case IDs: ' + str(path))
    return result


def selection(students):
    return sorted(students, key=lambda s: hashlib.sha256(('123:' + s).encode()).hexdigest())[:60]


def llm_is_a(case_id):
    return int(hashlib.sha256(('123:blind:' + case_id).encode()).hexdigest()[0], 16) % 2 == 0


def csv_file(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(text):
    # Do not strip, shorten, repair or reformat the saved answer text.
    return '<div class="answer">' + html.escape(text if text is not None else '[No saved explanation available]') + '</div>'


def page(title, instructions, sections):
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>' + html.escape(title) + '</title><style>'
            'body{font:17px/1.6 Arial,sans-serif;max-width:950px;margin:32px auto;padding:0 20px;color:#222}'
            'section{border-top:2px solid #ddd;margin-top:40px;padding-top:20px}'
            '.answer{white-space:pre-wrap;background:#f8f5ef;padding:20px;border:1px solid #ddd}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}'
            '@media print{section{break-before:page}.answer{background:white}}'
            '</style><body><h1>' + html.escape(title) + '</h1>' + instructions + ''.join(sections) + '</body></html>')


def build(experiment, collection, output):
    experiment, collection, output = experiment.resolve(), collection.resolve(), output.resolve()
    require(not output.exists(), 'Output already exists; use a NEW folder.')
    require(not output.is_relative_to(experiment) and not output.is_relative_to(collection),
            'Output must be separate from the frozen experiment and collection.')
    protocol_path = experiment / 'protocol.json'
    require(digest(protocol_path) == EXPECTED_PROTOCOL, 'Protocol hash does not match the reviewed experiment.')
    protocol = read(protocol_path)
    ledger = read(experiment / 'frozen_sha256.json')
    for name, expected in ledger.items():
        path = (experiment / name).resolve()
        require(path.is_relative_to(experiment), 'Invalid ledger path.')
        require(digest(path) == expected, 'Frozen file changed: ' + name)
    source = {
        'answers': collection / 'llm_explanations.jsonl',
        'prompts': experiment / 'all_prompts.jsonl',
        'templates': experiment / 'template_explanations.jsonl',
    }
    answers, prompts, templates = (indexed(source[k]) for k in ('answers', 'prompts', 'templates'))
    require(len(answers) == len(prompts) == len(templates) == 1326, 'Expected 1326 records in each source.')
    require(answers.keys() == prompts.keys() == templates.keys(), 'Case sets differ.')
    students = {r['student_key'] for r in prompts.values()}
    require(len(students) == 442, 'Expected 442 students.')
    selected = read(experiment / 'human_review_student_keys.json')
    require(selected == selection(students), 'Saved student selection does not match the frozen hash rule.')
    require(protocol['human_review_plan']['student_sample_size'] == 60, 'Unexpected review size.')
    for key, prompt in prompts.items():
        for row in (answers[key], templates[key]):
            require(row['student_key'] == prompt['student_key'] and row['stage'] == prompt['stage'], 'Identity mismatch: ' + key)
        require(answers[key].get('protocol_sha256') == EXPECTED_PROTOCOL, 'Answer protocol mismatch: ' + key)
    for student in students:
        require({r['stage'] for r in prompts.values() if r['student_key'] == student} == set(STAGES), 'Incomplete stages: ' + student)
    ids = [key for student in selected for stage in STAGES for key, r in prompts.items()
           if r['student_key'] == student and r['stage'] == stage]
    require(len(ids) == 180, 'Expected 180 paired cases.')
    reviewer = output / 'reviewer_files'
    private = output / 'private_DO_NOT_SHARE'
    reviewer.mkdir(parents=True, exist_ok=False)
    private.mkdir()
    clarity_sections, factual_sections, mapping, ratings = [], [], [], []
    for number, key in enumerate(ids, 1):
        review_id = f'PAIR_{number:03d}'
        method_a = 'llm' if llm_is_a(key) else 'template'
        pair = {'llm': answers[key].get('text'), 'template': templates[key]['text']}
        a, b = pair[method_a], pair['template' if method_a == 'llm' else 'llm']
        mapping.append({'review_id': review_id, 'case_id': key, 'student_key': prompts[key]['student_key'],
                        'stage': prompts[key]['stage'], 'A_method': method_a,
                        'B_method': 'template' if method_a == 'llm' else 'llm',
                        'llm_status': answers[key]['status']})
        heading = '<section><h2>' + review_id + '</h2>'
        pair_html = '<h3>Explanation A</h3>' + display(a) + '<h3>Explanation B</h3>' + display(b)
        clarity_sections.append(heading + pair_html + '</section>')
        # Evidence only; never expose the system prompt, model method, or true outcomes.
        evidence = prompts[key]['evidence']
        keys = ('feature_window', 'outcome_horizon', 'predicted_class', 'probabilities_percent',
                'probability_note', 'direction_scope', 'financial_timing')
        table = '<table>' + ''.join('<tr><th>' + html.escape(k.replace('_', ' ')) + '</th><td>' +
                                    html.escape(str(evidence[k])) + '</td></tr>' for k in keys if k in evidence) + '</table>'
        table += '<h3>Supplied contributors</h3><table><tr><th>Feature</th><th>Recorded</th><th>Decoded</th><th>Direction</th></tr>'
        for c in evidence['contributors']:
            table += '<tr>' + ''.join('<td>' + html.escape(str(c.get(k, ''))) + '</td>' for k in
                                     ('feature', 'recorded_value', 'decoded_value', 'direction')) + '</tr>'
        factual_sections.append(heading + table + '</table>' + pair_html + '</section>')
        ratings.append({'review_id': review_id})
    anchors = '<h2>Rating scales</h2>'
    for metric in ('clarity', 'usefulness'):
        anchors += '<h3>' + metric.title() + '</h3><ul>' + ''.join('<li>' + k + ': ' + html.escape(v) + '</li>' for k, v in
                protocol['human_review_plan'][metric + '_anchors'].items()) + '</ul>'
    instructions = ('<p>Work independently. Read both unchanged explanations. Score each separately, then record preference A/B/tie. '
                    'Judge understanding of the model output, not whether you agree with its prediction. Do not guess the method. '
                    'Complete clarity/usefulness ratings before opening the factual-audit file. This page does not save ratings; '
                    'enter them in your assigned CSV. Save the CSV and keep a backup.</p>' + anchors)
    (reviewer / '01_explanations.html').write_text(page('Explanation review', instructions, clarity_sections), encoding='utf-8')
    factual_instructions = ('<p>Open only after completing clarity/usefulness ratings. Compare each answer against the supplied evidence. '
        'For every flag enter 0 = absent, 1 = present, or U = uncertain. Blank means not reviewed. '
        'A contributor omission is a missing selected feature; value/direction omissions are missing supplied details. '
        'Numerical errors are changed numbers or class assignments. Unsupported claims are assertions not supported by this evidence. '
        'Causal/probability confusion includes treating raw-score directions as probability changes or causal effects. '
        'Missing safeguards include missing uncertainty, non-causal, partial-explanation or Dropout-only scope qualifications. '
        'Do not count harmless punctuation, rounding or synonymous wording as factual errors. Explain every 1 or U in notes. '
        'Empty or failed answers are not evidence of factual correctness. Do not consult actual student outcomes.</p>')
    (reviewer / '02_factual_audit.html').write_text(page('Evidence fidelity review', factual_instructions, factual_sections), encoding='utf-8')
    fields = ['review_id', 'rater_id', 'clarity_A', 'clarity_B', 'usefulness_A', 'usefulness_B', 'preference', 'rating_notes']
    fields += [f'{flag}_{arm}' for arm in ('A', 'B') for flag in FLAGS] + ['factual_notes']
    for rater in ('R1', 'R2'):
        csv_file(reviewer / f'ratings_{rater}.csv', fields, [dict(row, rater_id=rater) for row in ratings])
    csv_file(private / 'method_key.csv', list(mapping[0]), mapping)
    manifest = {'protocol_sha256': EXPECTED_PROTOCOL, 'students': 60, 'pairs': 180,
                'source_sha256': {k: digest(v) for k, v in source.items()},
                'pack_files_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
                'limitations': 'Labels are hidden, but writing style may reveal methods. No ratings are inferred or fabricated.'}
    (private / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (reviewer / 'START_HERE.txt').write_text(
        'Review independently; do not discuss scores until both reviewers finish.\n'
        '1. Open 01_explanations.html in a browser. Enter clarity/usefulness 1-5 and preference A/B/tie in your assigned CSV.\n'
        '2. Then open 02_factual_audit.html. Complete the separate factual flags and notes.\n'
        '3. Return your completed CSV. Blank cells mean missing, not zero. Do not edit explanation text.\n'
        'Use R1 or R2 as assigned. Additional reviewers require their own blank copy with a distinct ID.\n'
        'There are 180 pairs. Take breaks. Do not use an LLM to assign human-review scores.\n', encoding='utf-8')
    print('SUCCESS: 60 frozen students, 180 blinded pairs, two blank rating sheets.')
    print('Share ONLY:', reviewer)
    print('Keep PRIVATE:', private)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.experiment, args.collection, args.output)
