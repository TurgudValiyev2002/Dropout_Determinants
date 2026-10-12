"""Post-hoc automated comparison; never a substitute for human usefulness ratings.

Inputs are read-only. All paired cases retained. No outcome access or inference.
Requires numpy, pandas, matplotlib. No GPU required.
"""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROTOCOL_HASH = 'b19e9f81939fc4ddf9cba902388ba02bdbea456e11b4ba70fb7c9137a46f20a0'
STAGES = ['A_pre_enrollment', 'B_first_semester', 'C_second_semester']
CLOSING = 'These are uncalibrated estimates, not certainties or causes. Financial-record timing is uncertain.'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    result = {r['case_id']: r for r in rows}
    check(len(result) == len(rows), 'Duplicate case IDs: ' + str(path))
    return result


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def probability_screen(text, evidence):
    intro = text.split('\n\n')[0]
    states, details = [], {}
    for label, expected in evidence['probabilities_percent'].items():
        patterns = [
            rf'\b{label}\b\s*(?:(?:,\s*(?=with\b)|[:(])\s*)?'
            rf'(?:(?:with|has|is|at|a|probability|of|labelled|estimated|supplied|follows|stands)\s*){{0,6}}(\d+(?:\.\d+)?)\s*%',
            rf'(\d+(?:\.\d+)?)\s*%\s*(?:(?:for|of)\s+)?{label}\b',
        ]
        values = sorted({float(v) for pattern in patterns for v in re.findall(pattern, intro, re.I)})
        state = 'unresolved' if not values else ('matched' if all(abs(v - expected) < 1e-8 for v in values) else 'contradiction_screen')
        states.append(state)
        details[label] = {'expected': expected, 'extracted': values, 'screen_status': state}
    overall = 'contradiction_screen' if 'contradiction_screen' in states else ('unresolved' if 'unresolved' in states else 'matched')
    return overall, details


def measure(text, evidence):
    words = len(text.split())
    normalized = ' ' + normalize(text) + ' '
    features = [c['feature'] for c in evidence['contributors']]
    mentions = [f for f in features if ' ' + normalize(f) + ' ' in normalized]
    probability_status, probability_details = probability_screen(text, evidence)
    return {
        'word_count': words, 'over_120': int(words > 120),
        'exact_closing': int(text.rstrip().endswith(CLOSING)),
        'uncalibrated_word_present': int(bool(re.search(r'\buncalibrated\b', text, re.I))),
        'financial_warning_exact_present': int('financial-record timing is uncertain' in text.lower()),
        'raw_score_phrase_present': int(bool(re.search(r'raw[- ]score|raw score', text, re.I))),
        'partial_explanation_phrase_present': int(bool(re.search(r'not (?:the |a )?(?:full|complete) explanation|partial explanation', text, re.I))),
        'selected_feature_name_mention_fraction': len(mentions) / len(features) if features else np.nan,
        'probability_screen': probability_status,
        'probability_details': json.dumps(probability_details, ensure_ascii=False),
        'unmatched_feature_names': json.dumps([f for f in features if f not in mentions]),
    }


def template(e):
    return (f"{e['feature_window']} specification; outcome at the end of the normal course duration. "
            f"Highest-probability category: {e['predicted_class']}. "
            + ' '.join(f"{label} {e['probabilities_percent'][label]:.1f}%;" if label != 'Graduate'
                       else f"{label} {e['probabilities_percent'][label]:.1f}%." for label in ('Dropout', 'Enrolled', 'Graduate'))
            + ' Selected contributors, not the full explanation: '
            + ''.join(f"{c['feature']} ({c['decoded_value']}) {c['direction']} the Dropout raw score relative to its baseline. " for c in e['contributors'])
            + CLOSING)


def paired_stats(frame):
    # Each row is a student, not an explanation. Keep three stages together.
    rng = np.random.default_rng(123)
    metrics = ['word_count', 'over_120', 'exact_closing']
    results = []
    for metric in metrics:
        means = frame.groupby(['student_key', 'method'])[metric].mean().unstack('method').sort_index()
        differences = (means['llm'] - means['template']).to_numpy()
        boot = differences[rng.integers(0, len(differences), size=(2000, len(differences)))].mean(axis=1)
        positive = int((differences > 1e-12).sum())
        negative = int((differences < -1e-12).sum())
        nonzero = positive + negative
        # Exact sign test: tests direction balance among nonzero student differences,
        # NOT the null of zero mean. Ties explicitly reported, never hidden.
        p = min(1.0, 2 * sum(math.comb(nonzero, k) for k in range(min(positive, negative) + 1)) / (2 ** nonzero)) if nonzero else 1.0
        low, high = np.quantile(boot, [.025, .975])
        results.append({'metric': metric, 'students': len(differences),
                        'llm_mean': means['llm'].mean(), 'template_mean': means['template'].mean(),
                        'mean_difference_llm_minus_template': differences.mean(),
                        'bootstrap_ci_low': low, 'bootstrap_ci_high': high,
                        'positive_students': positive, 'negative_students': negative,
                        'tied_students': len(differences) - nonzero, 'sign_test_p': p})
    # Holm correction across the three post-hoc tests, not across arbitrary screens.
    order = np.argsort([r['sign_test_p'] for r in results])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(results) - rank) * results[index]['sign_test_p']))
        results[index]['sign_test_p_holm'] = running
    return pd.DataFrame(results)


def plot_results(frame, output):
    # Contract: static research exports, source per_case_metrics.csv.
    # Distribution: six boxplots; grain=explanation, descriptive only (not independent).
    # Compliance: stage-specific fractions; each denominator=442; bars start at zero.
    # Single blue root + neutral/open fill distinguishes methods in grayscale.
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#306C9E', '#FFFFFF']
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    groups = [frame[(frame.stage == stage) & (frame.method == method)].word_count.to_numpy()
              for stage in STAGES for method in ('llm', 'template')]
    boxes = ax.boxplot(groups, positions=[1, 1.7, 3, 3.7, 5, 5.7], widths=.5, patch_artist=True,
                       medianprops={'color': '#222222'}, flierprops={'markersize': 2, 'alpha': .3})
    for index, box in enumerate(boxes['boxes']):
        box.set_facecolor(colors[index % 2])
        box.set_edgecolor('#222222')
    ax.axhline(120, color='#555555', linestyle='--', linewidth=1, label='120-word limit')
    ax.set_xticks([1, 1.7, 3, 3.7, 5, 5.7], ['LLM', 'Template'] * 3)
    ax.set_ylabel('Words per explanation')
    ax.set_ylim(bottom=0)
    ax.set_title('Explanation length by feature window\n442 matched students per window; boxplots are descriptive')
    for pos, label in zip([1.35, 3.35, 5.35], ['Pre-enrollment', 'First semester', 'Second semester']):
        ax.text(pos, -.16, label, transform=ax.get_xaxis_transform(), ha='center')
    ax.legend(loc='upper right')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('length_by_stage.' + suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    x = np.arange(3)
    for i, method in enumerate(('llm', 'template')):
        rates = [100 * frame[(frame.stage == s) & (frame.method == method)].over_120.mean() for s in STAGES]
        bars = ax.bar(x + (i - .5) * .34, rates, .34, color=colors[i], edgecolor='#222222', label=method.upper() if i == 0 else 'Template')
        ax.bar_label(bars, labels=[f'{r:.1f}%' for r in rates], padding=3)
    ax.set_xticks(x, ['Pre-enrollment', 'First semester', 'Second semester'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Explanations exceeding 120 words (%)')
    ax.set_title('Word-limit noncompliance\n442 explanations per method in each window')
    ax.legend()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('word_limit_by_stage.' + suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(args):
    experiment, collection, output = args.experiment.resolve(), args.collection.resolve(), args.output.resolve()
    check(not output.exists(), 'Use a NEW output directory; refusing to overwrite.')
    check(not output.is_relative_to(experiment) and not output.is_relative_to(collection), 'Use a separate output directory.')
    check(sha(experiment / 'protocol.json') == PROTOCOL_HASH, 'Original protocol hash mismatch.')
    for name, expected in read(experiment / 'frozen_sha256.json').items():
        path = (experiment / name).resolve()
        check(path.is_relative_to(experiment) and sha(path) == expected, 'Frozen file changed: ' + name)
    files = {'llm': collection / 'llm_explanations.jsonl', 'prompts': experiment / 'all_prompts.jsonl',
             'template': experiment / 'template_explanations.jsonl'}
    answers, prompts, templates = (load(files[k]) for k in ('llm', 'prompts', 'template'))
    check(len(answers) == len(prompts) == len(templates) == 1326, 'Expected 1326 cases per source.')
    check(answers.keys() == prompts.keys() == templates.keys(), 'Case sets do not match.')
    rows = []
    for key, prompt in prompts.items():
        e = prompt['evidence']
        check(templates[key]['text'] == template(e), 'Template reconstruction mismatch: ' + key)
        check(answers[key].get('protocol_sha256') == PROTOCOL_HASH, 'Answer protocol mismatch.')
        for method, record in [('llm', answers[key]), ('template', templates[key])]:
            check(all(record[k] == prompt[k] for k in ('student_key', 'stage')), 'Identity mismatch: ' + key)
            text = record.get('text') or ''
            rows.append(dict(case_id=key, student_key=prompt['student_key'], stage=prompt['stage'], method=method,
                             status=record.get('status', 'fixed_template'), empty=int(not text.strip()),
                             token_limit_flag=record.get('token_limit_reached') if method == 'llm' else None,
                             **measure(text, e)))
    frame = pd.DataFrame(rows)
    check(frame.student_key.nunique() == 442, 'Expected 442 students.')
    check(set(frame.stage) == set(STAGES), 'Unexpected stages.')
    check(frame.groupby(['student_key', 'method']).size().eq(3).all(), 'Incomplete paired stages.')
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / 'per_case_metrics.csv', index=False)
    summary_rows = []
    for stage in ['ALL'] + STAGES:
        for method in ('llm', 'template'):
            subset = frame[(frame.method == method) & ((frame.stage == stage) if stage != 'ALL' else True)]
            summary_rows.append(dict(stage=stage, method=method, explanations=len(subset), students=subset.student_key.nunique(),
                                     mean_words=subset.word_count.mean(), median_words=subset.word_count.median(),
                                     max_words=subset.word_count.max(), over_120_n=int(subset.over_120.sum()),
                                     over_120_percent=100*subset.over_120.mean(), exact_closing_n=int(subset.exact_closing.sum()),
                                     empty_n=int(subset['empty'].sum())))
    pd.DataFrame(summary_rows).to_csv(output / 'summary_by_stage.csv', index=False)
    tests = paired_stats(frame)
    tests.to_csv(output / 'paired_statistics.csv', index=False)
    screens = frame.groupby(['stage', 'method', 'probability_screen']).size().rename('explanations').reset_index()
    screens.to_csv(output / 'probability_screen_summary.csv', index=False)
    frame[(frame.probability_screen != 'matched') | (frame.selected_feature_name_mention_fraction < 1)].to_csv(
        output / 'unresolved_screens.csv', index=False)
    plot_results(frame, output)
    limits = (
        'Post-hoc exploratory automated evaluation chosen after inspection of the test explanations. '
        'The original human-review protocol remains unchanged; no human ratings were collected. '
        'No clarity, usefulness, educational benefit or semantic factual-accuracy claim. '
        'Probability extraction and feature-name mentions are lexical screens, not semantic validation; '
        'unresolved cases remain in denominators, not counted as proven errors or silently dropped. '
        'Decoded-value and direction fidelity are not established by these screens. '
        'Tests use 442 student-level means, with all three stages retained. '
        '2000 paired student bootstrap resamples, seed 123, percentile 95% intervals for mean differences. '
        'Exact two-sided sign tests evaluate positive/negative balance among nonzero student differences, NOT zero mean; '
        'ties are reported. Holm correction across three exploratory tests. '
        'Inference assumes students are independent draws from a relevant population; intervals do not establish external validity. '
        'Sparse failures can yield unstable percentile intervals. Shorter text is not automatically clearer. '
        'No inferential tests on lexical screens. Failed/empty outputs, if any, retain zero word counts and must be reported separately.'
    )
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'analysis_type': 'post_hoc_automated',
                'original_protocol_sha256': PROTOCOL_HASH, 'script_sha256': sha(Path(__file__)),
                'sources_sha256': {k: sha(v) for k, v in files.items()}, 'limitations': limits,
                'software': {'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__}}
    (output / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    (output / 'READ_ME.txt').write_text(limits + '\n\nRates are fractions in paired_statistics.csv; multiply by 100 for percentage points.\n'
        'Matched probability screen means all three labelled percentages were extracted and matched; it does not certify the whole answer.\n'
        'per_case_metrics.csv contains lexical warning presence and feature-name mention measures for transparent inspection.\n', encoding='utf8')
    print('SUCCESS: 1326 matched cases, 442 students; post-hoc automated evaluation only.')
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print('\nPAIRED STATISTICS\n' + tests.to_string(index=False))
    print('\nSaved to:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
