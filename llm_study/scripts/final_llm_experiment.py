"""Frozen three-stage explanation experiment. Local inputs only; no training.

prepare: build all test prompts/templates and freeze protocol before inference.
run: process one of six chunks, checkpointing each case; --resume skips saved cases.
collect: assemble every expected case, including failed or unattempted cases.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

STAGES = {'A_pre_enrollment': 'Pre-enrollment', 'B_first_semester': 'First-semester', 'C_second_semester': 'Second-semester'}
SETTINGS = {'enable_thinking': False, 'do_sample': False, 'max_new_tokens': 512, 'seed': 123,
            'dtype': 'bfloat16', 'attention': 'eager', 'device_map': 'balanced', 'word_limit': 120}
CLOSING = 'These are uncalibrated estimates, not certainties or causes. Financial-record timing is uncertain.'
SYSTEM = '''Explain the supplied student-outcome model evidence to a university adviser in clear, respectful English.
Use only the supplied facts. Do not invent motives, personal circumstances, category meanings, interventions, or comparisons such as high/low grades or only/few approved units. Preserve age AT ENROLLMENT and recorded historical values; do not call them current.
Write at most 120 words. Follow this order:
1. State the feature window, the highest-probability category, and ALL THREE labelled probabilities. State that the predicted outcome is at the END OF THE NORMAL COURSE DURATION, not at the end of a semester. Say highest-probability category, never that a student WILL graduate, remain enrolled, or drop out.
2. Describe EVERY listed contributor, with its supplied recorded/decoded value and supplied direction. Do not omit any contributor. These directions refer ONLY to the Dropout model raw score relative to its SHAP baseline. They are not probability changes, causal effects, or effects of changing a feature. They do not explain why Graduate rather than Enrolled wins. Do not infer a trend from one record or stereotype sensitive groups. Say these are selected contributors, not the full explanation. Do not invent any raw score, baseline, arithmetic, or percentage-point effect.
3. End with this EXACT sentence pair: These are uncalibrated estimates, not certainties or causes. Financial-record timing is uncertain.
Tuition fees up to date = no means not up to date; do not strengthen this to never paid or inability to pay. Unknown/blank decoded categories remain unknown/blank. A zero grade is the recorded zero, not proof of no attendance. Application order is a recorded rank, not a code to reinterpret.
Do not give recommendations. Do not say predictions cannot predict outcomes: they are uncertain predictions, not established outcomes. No headings, lists, or raw-score numbers are needed.'''


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(value):
    return hashlib.sha256(('123:' + value).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def jsonl(path, items):
    with path.open('w', encoding='utf-8') as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')


def load_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def safe(value):
    if isinstance(value, dict):
        check(not (set(value) & {'StudentID', 'actual_outcome', 'Target', 'DropoutBinary', 'DropoutLabel', 'y'}), 'Forbidden identifier/outcome key')
        for v in value.values():
            safe(v)
    elif isinstance(value, list):
        for v in value:
            safe(v)


def decode(fact, book):
    feature, raw = fact['feature'], fact['recorded_value']
    entry = book['features'][feature]
    status = 'numeric'
    if raw is None:
        value, status = 'not recorded', 'missing'
    elif fact['value_type'] == 'categorical_code':
        check(float(raw).is_integer(), 'Noninteger category')
        code = str(int(raw))
        if feature == 'Application order':
            value, status = f'recorded rank {code}', 'documented_ordinal'
        else:
            value = entry['code_labels'].get(code, f'code {code}, meaning unverified')
            status = 'verified_code' if code in entry['code_labels'] else 'unresolved_code'
    elif 'grade' in feature.lower():
        value = f"{raw:g} on a 0–{20 if 'sem' in feature else 200} scale"
    else:
        value = f'{raw:g}' if isinstance(raw, (int, float)) else str(raw)
    return {'feature': feature, 'recorded_value': raw, 'decoded_value': value, 'decoding_status': status,
            'direction': 'raises' if fact['dropout_raw_score_contribution'] > 0 else 'lowers'}


def template(evidence):
    p = evidence['probabilities_percent']
    result = (f"{evidence['feature_window']} specification; outcome at the end of the normal course duration. "
              f"Highest-probability category: {evidence['predicted_class']}. "
              f"Dropout {p['Dropout']:.1f}%; Enrolled {p['Enrolled']:.1f}%; Graduate {p['Graduate']:.1f}%. ")
    result += 'Selected contributors, not the full explanation: '
    for fact in evidence['contributors']:
        result += f"{fact['feature']} ({fact['decoded_value']}) {fact['direction']} the Dropout raw score relative to its baseline. "
    return result + CLOSING


def prepare(args):
    root, book_path, out = args.evidence_root.resolve(), args.codebook.resolve(), args.experiment.resolve()
    check(not out.exists() and not out.is_relative_to(root) and not out.is_relative_to(book_path.parent), 'Use a NEW directory outside inputs')
    source = root / 'llm_inputs/test_evidence.jsonl'
    ledger = {k.replace('\\', '/'): v for k, v in read(root / 'output_sha256.json').items()}
    check(sha(source) == ledger['llm_inputs/test_evidence.jsonl'], 'Test evidence hash mismatch')
    check(sha(root / 'manifest.json') == ledger['manifest.json'], 'Manifest hash mismatch')
    book, manifest = read(book_path), read(root / 'manifest.json')
    input_hashes = set(manifest['input_sha256'].values())
    check(book['local_data_sha256'] in input_hashes and book['local_metadata_sha256'] in input_hashes, 'Codebook/evidence source mismatch')
    records = load_lines(source)
    check(len(records) == 1326 and len({r['case_id'] for r in records}) == 1326, 'Expected 1326 unique test cases')
    groups, prompts, baselines, audits = {}, [], [], []
    for record in records:
        safe(record)
        matched = re.fullmatch(r'test_(\d{4})_(A_pre_enrollment|B_first_semester|C_second_semester)', record['case_id'])
        check(matched is not None, 'Unexpected test case ID')
        student, stage = 'test_' + matched[1], matched[2]
        groups.setdefault(student, set()).add(stage)
        facts = record['predictor_evidence']
        check(len({f['feature'] for f in facts}) == len(facts), 'Duplicate predictor names')
        check(all(math.isfinite(f['dropout_raw_score_contribution']) for f in facts), 'Invalid SHAP value')
        check(abs(sum(f['dropout_raw_score_contribution'] for f in facts) + record['dropout_shap_baseline'] - record['dropout_raw_score']) < 1e-6, 'SHAP reconstruction failed')
        pos = sorted((f for f in facts if f['dropout_raw_score_contribution'] > 0), key=lambda f: (-f['dropout_raw_score_contribution'], f['feature']))[:2]
        neg = sorted((f for f in facts if f['dropout_raw_score_contribution'] < 0), key=lambda f: (f['dropout_raw_score_contribution'], f['feature']))[:1]
        chosen = pos + neg
        probabilities = record['model_probabilities']
        check(set(probabilities) == {'Dropout', 'Enrolled', 'Graduate'} and all(0 <= v <= 1 for v in probabilities.values()) and abs(sum(probabilities.values()) - 1) < 1e-8, 'Invalid probabilities')
        check(record['predicted_class'] == max(probabilities, key=probabilities.get), 'Predicted class mismatch')
        evidence = {'feature_window': STAGES[stage], 'outcome_horizon': 'End of normal course duration',
            'predicted_class': record['predicted_class'], 'probabilities_percent': {k: round(100 * probabilities[k], 1) for k in ('Dropout', 'Enrolled', 'Graduate')},
            'probability_note': 'Rounded independently; totals may differ slightly from 100%.',
            'contributors': [decode(f, book) for f in chosen],
            'direction_scope': 'Selected Dropout raw-score contributors only, not probability effects or a complete explanation',
            'financial_timing': 'Measurement timing uncertain; nominal feature windows do not prove prospective availability'}
        text = template(evidence)
        check(len(text.split()) <= SETTINGS['word_limit'], f"Template exceeds word limit: {record['case_id']}; do not run until protocol is resolved")
        prompts.append({'case_id': record['case_id'], 'student_key': student, 'stage': stage, 'evidence': evidence,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'Explain all the following evidence.\n' + json.dumps(evidence, ensure_ascii=False)}]})
        baselines.append({'case_id': record['case_id'], 'student_key': student, 'stage': stage, 'method': 'fixed_template_v2', 'text': text, 'word_count': len(text.split())})
        audits.append({'case_id': record['case_id'], 'selected_shap': chosen, 'raw_score': record['dropout_raw_score'], 'baseline': record['dropout_shap_baseline'],
                       'unshown_contributions_sum': sum(f['dropout_raw_score_contribution'] for f in facts if f not in chosen)})
    check(len(groups) == 442 and all(stages == set(STAGES) for stages in groups.values()), 'Each of 442 students must occur in all three stages')
    prompts.sort(key=lambda r: rank(r['case_id']))
    by_id = {r['case_id']: r for r in baselines}
    human_students = sorted(groups, key=rank)[:60]
    protocol = {'version': 'final_v2', 'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': SETTINGS,
        'question': 'Does an LLM produce clearer, useful explanations than a fixed template while preserving the same model evidence?',
        'n_students': 442, 'n_explanations_per_method': 1326, 'stages': STAGES, 'chunks': 6, 'cases_per_chunk': 221,
        'selection': 'All test students, all three stages; no case exclusion by outcome, confidence, sensitive features or generation quality.',
        'contributors': 'Up to two largest positive and one largest-magnitude negative Dropout SHAP contributions; ties by original feature name. Same decoded facts in both arms. Baseline/final raw score/remainder retained only in audit files, not writer input.',
        'failure_policy': 'One generation attempt per case. Save empty, overlength, truncated and failed responses. Resume only cases with no saved record after technical interruption; never retry an existing saved case or select among outputs.',
        'human_review_plan': {'student_sample_size': 60, 'selection': 'Smallest SHA256(123:student_key), fixed before inference, all three stages: 180 pairs',
            'raters_required': 'At least two independent educational advisers or qualified education researchers; not yet recruited/confirmed',
            'blinding': 'Method labels hidden, randomized A/B order per pair; audit integrity separately with supplied evidence. Keep mapping private.',
            'ratings': 'Clarity and practical usefulness separately on anchored 1–5 scales; preferred explanation A/B/tie; contributor/value/direction omissions, numerical errors, unsupported claims, causal/probability confusion and missing safeguards as separate flags.',
            'clarity_anchors': {'1': 'Very difficult to understand or misleading', '2': 'Substantial interpretation needed', '3': 'Understandable with some effort', '4': 'Clear with minor issues', '5': 'Immediately clear and unambiguous'},
            'usefulness_anchors': {'1': 'Not useful for understanding this model output', '2': 'Limited useful information', '3': 'Moderately useful', '4': 'Useful for interpreting the supplied model evidence', '5': 'Highly useful while preserving limitations; not a rating of intervention effectiveness'},
            'A_B_order_rule': "LLM is A if the first hexadecimal digit of SHA256('123:blind:'+case_id) is even; otherwise template is A. Mapping retained privately, not given to raters.",
            'primary_endpoint': 'Mean paired LLM-minus-template clarity difference on preselected 60 students, averaging raters then three stages within student.',
            'secondary_endpoints': 'Usefulness difference, preference/tie proportions, error/omission rates, per-stage results and subgroup descriptive checks.',
            'inference': '2000 paired student-cluster bootstrap resamples, seed 123, percentile 95% intervals. Resample students, retaining their stages and method pairs. Stage-specific results secondary; no independent-sample analysis of 1326 records.',
            'missingness': 'Report recruitment/completion and every missing rating or generation. Never call a missing/failed output correct. Primary clarity comparison uses students with all planned paired ratings; report denominator and sensitivity summaries for all available pairs and generation failure rates.'},
        'limits': ['Prompt revised on validation pilot only; no further pilot requested. No guarantee of final compliance.', 'The prediction benchmark/test set was previously examined; this is held-out explanation evaluation relative to prompt development, not a new independent predictive validation.',
                   'This is verbalization of deterministically decoded and direction-labelled evidence, not an LLM test of inferring SHAP signs from scratch. Public-dataset exposure during LLM pretraining cannot be ruled out.',
                   'No claims of causality, improved retention, intervention effectiveness, predictive accuracy gains, calibrated risk or distance-education validity.', 'Automatic text checks are screening, not proof of semantic correctness; human evaluation is required for clarity/usefulness claims.'],
        'inputs_sha256': {'test_evidence': sha(source), 'codebook': sha(book_path), 'manifest': sha(root / 'manifest.json')},
        'runner_sha256': sha(Path(__file__))}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'chunks').mkdir()
    (out / 'runs').mkdir()
    save(out / 'protocol.json', protocol)
    save(out / 'human_review_student_keys.json', human_students)
    (out / 'system_prompt_v2.txt').write_text(SYSTEM, encoding='utf-8')
    jsonl(out / 'all_prompts.jsonl', prompts)
    jsonl(out / 'template_explanations.jsonl', [by_id[r['case_id']] for r in prompts])
    jsonl(out / 'diagnostics_DO_NOT_PROMPT.jsonl', audits)
    for chunk in range(6):
        jsonl(out / 'chunks' / f'chunk_{chunk}.jsonl', prompts[chunk * 221:(chunk + 1) * 221])
    save(out / 'frozen_sha256.json', {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print('PREPARED: 442 students × 3 stages = 1326 prompts and 1326 matched templates; six chunks of 221.')
    print('Prompt/settings/protocol frozen. No LLM inference performed. Saved to', out)


def load_chunk(experiment, chunk):
    ledger = read(experiment / 'frozen_sha256.json')
    for name in ledger:
        check(sha(experiment / name) == ledger[name], 'Frozen input changed: ' + name)
    protocol = read(experiment / 'protocol.json')
    check(protocol['runner_sha256'] == sha(Path(__file__)) and protocol['settings'] == SETTINGS, 'Runner/settings changed after freeze')
    records = load_lines(experiment / 'chunks' / f'chunk_{chunk}.jsonl')
    check(len(records) == 221 and len({r['case_id'] for r in records}) == 221, 'Wrong chunk size/duplicates')
    for record in records:
        safe(record)
        check(record['stage'] in STAGES and record['case_id'].startswith('test_'), 'Invalid test case')
        check(json.loads(record['messages'][1]['content'].split('\n', 1)[1]) == record['evidence'], 'Message/evidence mismatch')
    return protocol, records, sha(experiment / 'protocol.json')


def run(args):
    experiment, model_path = args.experiment.resolve(), args.model.resolve()
    protocol, records, protocol_hash = load_chunk(experiment, args.chunk)
    if args.check_only:
        print(f'CHECK PASSED: frozen chunk {args.chunk}, 221 cases. No GPU execution or file writes.')
        return
    check(model_path.is_dir() and not experiment.is_relative_to(model_path), 'Invalid model/experiment path')
    out = experiment / 'runs' / f'chunk_{args.chunk}'
    if args.resume:
        check(out.is_dir(), 'No existing chunk to resume')
    else:
        out.mkdir(exist_ok=False)
    # Linux advisory lock releases automatically if Slurm terminates the job.
    import fcntl
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = out / 'cases'
        cases.mkdir(exist_ok=True)
        identity = {'protocol_sha256': protocol_hash, 'model_path': str(model_path),
                    'model_metadata_sha256': {p.name: sha(p) for p in [model_path / 'config.json', model_path / 'tokenizer_config.json', model_path / 'chat_template.jinja', model_path / 'model.safetensors.index.json'] if p.is_file()}}
        if args.resume:
            check(read(out / 'identity.json') == identity, 'Resume identity mismatch')
        else:
            save(out / 'identity.json', identity)
        pending = [r for r in records if not (cases / (r['case_id'] + '.json')).exists()]
        print(f'Chunk {args.chunk}: {len(pending)} pending; saved cases will not be regenerated.', flush=True)
        if not pending:
            return
        sys.dont_write_bytecode = True
        for variable, suffix in {'HF_HOME': 'hf', 'XDG_CACHE_HOME': 'xdg', 'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'extensions', 'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'NUMBA_CACHE_DIR': 'numba'}.items():
            cache = out / 'runtime_cache' / suffix
            cache.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(cache)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            check(torch.cuda.device_count() == 2, 'Exactly two allocated GPUs required')
            torch.manual_seed(123)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            print('Loading BF16 model across two GPUs ...', flush=True)
            started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                device_map='balanced', max_memory={0: '38GiB', 1: '38GiB', 'cpu': 0}, attn_implementation='eager')
            model.eval()
            placement = {k: str(v) for k, v in model.hf_device_map.items()}
            check(not any(v in {'cpu', 'disk', 'meta'} for v in placement.values()), 'Unexpected offload')
            invocation = {'started_utc': datetime.now(timezone.utc).isoformat(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                'settings': SETTINGS, 'load_seconds': time.perf_counter() - started, 'device_map': placement,
                'environment': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'accelerate')},
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(2)], 'python': sys.version, 'resume': args.resume}
            with (out / 'invocations.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(invocation) + '\n')
            device = model.get_input_embeddings().weight.device
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            deadline_start = time.monotonic()
            for number, record in enumerate(pending, 1):
                # Leave checkpoint margin before the two-hour partition limit.
                if time.monotonic() - deadline_start > 105 * 60:
                    print('Checkpoint stop before wall-time limit; resume this chunk without changing inputs.', flush=True)
                    break
                result = {'case_id': record['case_id'], 'student_key': record['student_key'], 'stage': record['stage'],
                    'method': 'nemotron_bf16_greedy_v2', 'status': 'failed', 'text': None,
                    'protocol_sha256': protocol_hash, 'chunk': args.chunk}
                failed = False
                try:
                    encoded = tokenizer.apply_chat_template(record['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt')
                    check(encoded['input_ids'].shape[-1] <= 4096, 'Prompt too long; no silent truncation')
                    rendered = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)
                    result['rendered_prompt_sha256'] = hashlib.sha256(rendered.encode()).hexdigest()
                    encoded = encoded.to(device)
                    for i in range(2):
                        torch.cuda.synchronize(i)
                        torch.cuda.reset_peak_memory_stats(i)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**encoded, max_new_tokens=512, do_sample=False, use_cache=True, pad_token_id=pad_id)
                    for i in range(2):
                        torch.cuda.synchronize(i)
                    elapsed = time.perf_counter() - started
                    ids = generated[0, encoded['input_ids'].shape[-1]:].tolist()
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    result.update(status='generated' if text.strip() else 'empty', text=text, word_count=len(text.split()),
                        over_word_limit=len(text.split()) > 120, token_limit_reached=len(ids) >= 512,
                        required_closing_exact=text.strip().endswith(CLOSING), generated_token_ids=ids, generated_tokens=len(ids),
                        prompt_tokens=int(encoded['input_ids'].shape[-1]), generation_seconds=elapsed,
                        peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(2)])
                except Exception:
                    result['error'] = traceback.format_exc()
                    failed = True
                final_path = cases / (record['case_id'] + '.json')
                check(not final_path.exists(), 'Refusing to overwrite a saved case')
                temporary = cases / (record['case_id'] + '.partial')
                save(temporary, result)
                os.replace(temporary, final_path)
                print(f"Chunk {args.chunk}: {number}/{len(pending)} {record['case_id']} {result['status']}", flush=True)
                if failed:
                    print('Stopping after saved failure; no automatic retry.', flush=True)
                    raise SystemExit(1)
            counts = Counter(read(p)['status'] for p in cases.glob('*.json'))
            print('CHUNK STATUS:', dict(counts), 'pending=', 221 - sum(counts.values()), flush=True)
        except Exception:
            with (out / 'technical_errors.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'error': traceback.format_exc()}) + '\n')
            raise


def collect(args):
    root, output = args.experiment.resolve(), args.output.resolve()
    check(not output.exists() and not output.is_relative_to(root), 'Use a NEW sibling collection directory')
    results = []
    for chunk in range(6):
        _, records, protocol_hash = load_chunk(root, chunk)
        for record in records:
            path = root / 'runs' / f'chunk_{chunk}' / 'cases' / (record['case_id'] + '.json')
            if path.is_file():
                result = read(path)
                check(result['case_id'] == record['case_id'] and result['protocol_sha256'] == protocol_hash and result['stage'] == record['stage'] and result['student_key'] == record['student_key'], 'Saved case identity mismatch')
            else:
                result = {k: record[k] for k in ('case_id', 'student_key', 'stage')}
                result.update(status='not_attempted', text=None, chunk=chunk)
            results.append(result)
    output.mkdir(parents=True, exist_ok=False)
    jsonl(output / 'llm_explanations.jsonl', results)
    summary = {'expected_records': 1326, 'students': 442, 'statuses': dict(Counter(r['status'] for r in results)),
        'by_stage': {stage: dict(Counter(r['status'] for r in results if r['stage'] == stage)) for stage in STAGES},
        'over_word_limit': sum(r.get('over_word_limit') is True for r in results),
        'token_limit_flags': sum(r.get('token_limit_reached') is True for r in results),
        'closing_mismatch_generated': sum(r['status'] == 'generated' and not r.get('required_closing_exact') for r in results),
        'quality_status': 'Not evaluated. Generation success and lexical compliance do not prove correctness or clarity.'}
    save(output / 'summary.json', summary)
    save(output / 'source_identity.json', {'protocol_sha256': sha(root / 'protocol.json'), 'frozen_ledger_sha256': sha(root / 'frozen_sha256.json')})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--evidence-root', type=Path, required=True)
    prep.add_argument('--codebook', type=Path, required=True)
    prep.add_argument('--experiment', type=Path, required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--experiment', type=Path, required=True)
    runner.add_argument('--model', type=Path, required=True)
    runner.add_argument('--chunk', type=int, choices=range(6), required=True)
    runner.add_argument('--resume', action='store_true')
    runner.add_argument('--check-only', action='store_true')
    collector = sub.add_parser('collect')
    collector.add_argument('--experiment', type=Path, required=True)
    collector.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'collect': collect}[args.command](args)


if __name__ == '__main__':
    main()
