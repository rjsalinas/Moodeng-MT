"""
normalize_pipeline.py

Usage:
- Place rules.json, lexica/*.json, regex_patterns.json, pipeline_config.json
  in the same folder (or edit PATHs below).
- Run: python normalize_pipeline.py
- The script will augment/validate rules and run a demo normalization on the `sample_texts` provided below.

Notes:
- This is a lightweight engine for prototyping. Replace predicate functions and morphological hooks
  with calamanCy calls when available.
"""

import json
import re
import os
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict

# ---------- Config / file paths ----------
BASE = Path(".")
RULES_FILE = BASE / "config/rules.json"
REGEX_FILE = BASE / "config/regex_patterns.json"
LEXICA_DIR = BASE / "config/lexica/lexica"
PIPE_CFG_FILE = BASE / "config/pipeline_config.json"

# ---------- Helpers to load files ----------
def load_json(p):
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)

def save_json(data, p):
    with open(p, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------- Load and merge multiple rules files ----------
def load_and_merge_rules(primary_rules_path: Path, secondary_rules_path: Path = None):
    """
    Load rules from primary (config/rules.json). Secondary path ignored (removed).
    Returns a dict with key 'rules'.
    """
    primary = load_json(primary_rules_path)
    merged = {"rules": []}
    id_to_rule = {}
    for r in primary.get("rules", []):
        rid = r.get("rule_id")
        if not rid:
            continue
        id_to_rule[rid] = r
    merged["rules"] = list(id_to_rule.values())
    return merged

# ---------- Basic morphological predicate stubs ----------
# If calamancy is installed, we will use it; otherwise, fallback heuristics.
try:
    import calamancy
    CALAMANCY_AVAILABLE = True
    nlp = calamancy.load("tl_calamancy_md-0.1.0")
    print("calamancy loaded")
except Exception:
    CALAMANCY_AVAILABLE = False
    nlp = None

def is_proper_name(token):
    if CALAMANCY_AVAILABLE:
        doc = nlp(token)
        # quick heuristic: if token labeled PROPN
        return any(tok.tag_ == "PROPN" or tok.ent_type_ == "PER" for tok in doc)
    # fallback: capitalized and not start-of-sentence heuristics
    return bool(token) and token[0].isupper()

def token_is_fil(token):
    # very simple fallback: presence of typical Tagalog function words or suffixes
    fil_indicators = ["ng", "ang", "si", "siya", "kayo", "ka", "ni"]
    t = token.lower()
    return any(t.endswith(s) or t == s for s in fil_indicators)

def morph_is_reduplication(token_a, token_b):
    # fallback heuristic: identical tokens -> reduplication
    if token_a is None or token_b is None:
        return False
    return token_a.lower() == token_b.lower()

# ---------- Augment rules: ensure keys exist with defaults ----------
def augment_rules(rules_data, pipeline_cfg):
    defaults = {
        "paper_rule": None,
        "direction": "context_dependent",
        "scope": "token",
        "canonical_choice": None,
        "rationale": None,
        "preconditions": [],
        "examples": [],
        "confidence_default": pipeline_cfg.get("default_confidence_if_missing", 0.75),
        "variation_index": "medium",
        "priority": 50,
        "active": True,
        "deprecated": False,
        "notes": None,
        "created_by": None,
        "created_date": None,
        "updated_by": None,
        "updated_date": None
    }
    changed = False
    for r in rules_data["rules"]:
        for k,v in defaults.items():
            if k not in r:
                r[k] = deepcopy(v)
                changed = True
        # normalize legacy_ids to exist
        if "legacy_ids" not in r or not isinstance(r["legacy_ids"], list):
            r["legacy_ids"] = []
    return rules_data, changed

# ---------- Merge lexica: load all lexica files and unify ----------
def load_and_merge_lexica(lexica_dir):
    lex = {}
    pdir = Path(lexica_dir)
    if not pdir.exists():
        return lex
    for p in pdir.glob("*.json"):
        d = load_json(p)
        for token, info in d.items():
            # Skip metadata blocks that annotate stage, description, etc.
            if token == "metadata":
                continue
            t = token.lower()
            if t not in lex:
                lex[t] = info
                continue
            # Merge dict/list gracefully
            existing = lex[t]
            # both dicts → keep higher confidence
            if isinstance(existing, dict) and isinstance(info, dict):
                if info.get("confidence", 0) > existing.get("confidence", 0):
                    lex[t] = info
                continue
            # coerce to list of senses and de-duplicate by (to, precondition)
            acc = []
            if isinstance(existing, dict):
                acc.append(existing)
            elif isinstance(existing, list):
                acc.extend(existing)
            if isinstance(info, dict):
                acc.append(info)
            elif isinstance(info, list):
                acc.extend(info)
            seen = set()
            merged = []
            for s in acc:
                key = (s.get("to"), s.get("precondition"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(s)
            lex[t] = merged
    return lex

# ---------- Validate rule uniqueness ----------
def validate_rules(rules_data):
    ids = set()
    legacy = {}
    errors = []
    for r in rules_data["rules"]:
        rid = r.get("rule_id")
        if not rid:
            errors.append("Missing rule_id in rule: {}".format(r))
            continue
        if rid in ids:
            errors.append(f"Duplicate rule_id: {rid}")
        ids.add(rid)
        for lid in r.get("legacy_ids", []):
            if lid in legacy and legacy[lid] != rid:
                errors.append(f"legacy_id {lid} reused by {rid} and {legacy[lid]}")
            legacy[lid] = rid
    return errors

# ---------- Simple tokenizer (whitespace + punctuation preserve) ----------
WORD_RE = re.compile(r"[^\s]+", re.UNICODE)
def simple_tokenize(text):
    return WORD_RE.findall(text)

# ---------- Load regex patterns ----------
def compile_regex_patterns(regex_file):
    raw = load_json(regex_file)
    compiled = {}
    for name, obj in raw.items():
        # Skip non-pattern metadata blocks
        if name == "metadata" or not isinstance(obj, dict) or "pattern" not in obj:
            continue
        try:
            compiled[name] = {
                "pattern": re.compile(obj["pattern"], flags=re.IGNORECASE),
                "replacement": obj.get("replacement"),
                "description": obj.get("description", "")
            }
        except re.error as e:
            print(f"Invalid regex for {name}: {e}")
    return compiled

# ---------- Sentence-level safe regex replacements ----------
def apply_sentence_regexes(text, compiled_regexes, applied_rules_list):
    # Optional lowercasing applied here if configured upstream; fallback handled in normalize_text
    # Expand di'ba/d'ba -> hindi ba
    rx = compiled_regexes.get("DIBA_EXPAND")
    if rx:
        new_text = rx["pattern"].sub(rx["replacement"], text)
        if new_text != text:
            applied_rules_list.append({"rule": "REGEX.DIBA_EXPAND", "confidence": 0.95})
            text = new_text
    # Add hyphen after pagka before word
    rx = compiled_regexes.get("AFFIX_HYPHEN_PAGKA")
    if rx:
        new_text = rx["pattern"].sub(rx["replacement"], text)
        if new_text != text:
            applied_rules_list.append({"rule": "REGEX.AFFIX_HYPHEN_PAGKA", "confidence": 0.95})
            text = new_text
    # nagreply -> nag-reply
    rx = compiled_regexes.get("NAGREPLY_HYPHEN")
    if rx:
        new_text = rx["pattern"].sub(rx["replacement"], text)
        if new_text != text:
            applied_rules_list.append({"rule": "REGEX.NAGREPLY_HYPHEN", "confidence": 0.95})
            text = new_text
    # Strip emojis and non-alphanumeric punctuation to spaces (conservative)
    rx = compiled_regexes.get("NON_ALNUM_STRIP")
    if rx:
        new_text = rx["pattern"].sub(rx["replacement"], text)
        if new_text != text:
            applied_rules_list.append({"rule": "REGEX.NON_ALNUM_STRIP", "confidence": 0.95})
            text = new_text
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Precondition evaluation (simple interpreter) ----------
def _atom_eval(atom: str, token: str, context: dict, lexica: dict):
    atom = atom.strip()
    m = re.match(r"^(prev_token|next_token|token_lower|prev_pos|next_pos)\s*==\s*['\"](.+)['\"]$", atom)
    if m:
        key, val = m.group(1), m.group(2)
        return (context.get(key) or "").lower() == val.lower()
    m = re.match(r"^(prev_token|next_token)\s*in\s*(\[[^\]]+\])$", atom)
    if m:
        key, list_literal = m.group(1), m.group(2)
        items = [x.strip().strip("'\"").lower() for x in re.split(r',\s*', list_literal.strip()[1:-1]) if x.strip()]
        return (context.get(key) or "").lower() in items
    if atom == "next_token_starts_with_vowel":
        nxt = (context.get("next_token") or "").lower()
        return bool(re.match(r'^[aeiou]', nxt))
    if atom == "token_starts_with_vowel":
        t = (context.get("token_lower") or "").lower()
        return bool(re.match(r'^[aeiou]', t))
    m = re.match(r"^token_length\s*==\s*(\d+)$", atom)
    if m:
        return int(m.group(1)) == int(context.get("token_length", 0))
    m = re.match(r"^next_matches\((['\"])(.+)\1\)$", atom)
    if m:
        rx = m.group(2)
        nxt = context.get("next_token") or ""
        return bool(re.search(rx, nxt))
    m = re.match(r"^prev_in_lexicon\((['\"])(.+)\1\)$", atom)
    if m:
        lexname = m.group(2)
        prev = (context.get("prev_token") or "").lower()
        return prev in lexica.get(lexname, {})
    # legacy atoms
    if atom == "not_proper_name":
        return not is_proper_name(token)
    if atom.startswith("lang in"):
        return token_is_fil(token)
    if atom == "morph.is_reduplication == true":
        prev = context.get("prev_token")
        if not prev:
            return False
        return morph_is_reduplication(prev, token)
    return False

def eval_precondition(expr: str, token: str, context: dict, lexica: dict):
    if not expr:
        return True
    expr = expr.strip()
    or_terms = [t.strip() for t in re.split(r'\s+or\s+', expr)]
    for term in or_terms:
        neg_term = False
        if term.startswith("not "):
            neg_term = True
            term = term[len("not "):].strip()
        and_atoms = [a.strip() for a in re.split(r'\s+and\s+', term)]
        all_and_true = True
        for atom in and_atoms:
            atom_neg = False
            if atom.startswith("not "):
                atom_neg = True
                atom = atom[len("not "):].strip()
            atom_result = _atom_eval(atom, token, context, lexica)
            if atom_neg:
                atom_result = not atom_result
            if not atom_result:
                all_and_true = False
                break
        term_result = not all_and_true if neg_term else all_and_true
        if term_result:
            return True
    return False

# ---------- Apply lexicon mappings first ----------
def apply_lexicon_contextual(token, context, lexica, default_conf=0.75):
    key = (token or "").lower()
    entry = lexica.get(key)
    if not entry:
        return None, 0.0, None
    if isinstance(entry, dict) and 'to' in entry:
        return entry['to'], float(entry.get('confidence', default_conf)), entry
    if isinstance(entry, list):
        best = None
        for sense in entry:
            pre = sense.get("precondition")
            ok = True if not pre else eval_precondition(pre, token, context, lexica)
            if ok:
                if best is None or sense.get("confidence", 0.0) > best.get("confidence", 0.0):
                    best = sense
        if best:
            return best.get("to"), float(best.get("confidence", default_conf)), best
        # fallback to highest-confidence overall
        fallback = max(entry, key=lambda s: s.get("confidence", 0.0))
        return fallback.get("to"), float(fallback.get("confidence", 0.1)), fallback
    return None, 0.0, None

def build_context(tokens, i, pos_tags=None):
    tok = tokens[i]
    prev = tokens[i-1] if i-1 >= 0 else None
    prev_prev = tokens[i-2] if i-2 >= 0 else None
    nxt = tokens[i+1] if i+1 < len(tokens) else None
    nxt2 = tokens[i+2] if i+2 < len(tokens) else None
    token_lower = (tok or "").lower()
    ctx = {
        "token": tok,
        "token_lower": token_lower,
        "token_length": len(tok) if tok else 0,
        "token_is_all_alpha": tok.isalpha() if tok else False,
        "token_has_digit": any(c.isdigit() for c in (tok or "")),
        "token_has_dash": "-" in (tok or ""),
        "token_starts_with_vowel": bool(re.match(r'^[aeiou]', token_lower)),
        "token_ends_with_vowel": bool(re.search(r'[aeiou]$', token_lower)) if token_lower else False,
        "prev_token": prev.lower() if prev else None,
        "prev_prev_token": prev_prev.lower() if prev_prev else None,
        "next_token": nxt.lower() if nxt else None,
        "next_next_token": nxt2.lower() if nxt2 else None,
        "is_sentence_start": i == 0,
        "is_sentence_end": i == len(tokens) - 1,
    }
    if pos_tags:
        ctx["prev_pos"] = pos_tags[i-1] if i-1 >= 0 else None
        ctx["next_pos"] = pos_tags[i+1] if i+1 < len(pos_tags) else None
        ctx["token_pos"] = pos_tags[i]
    return ctx

# ---------- Apply single rule (very conservative) ----------
def apply_rule_to_token(rule, token, prev_token, compiled_regexes, lexicon):
    applied = False
    new_token = token
    applied_rule_id = None
    rule_conf = rule.get("confidence_default", 0.75)
    scope = rule.get("scope","token")
    # token-level lexicon reference
    if isinstance(rule.get("pattern"), str) and rule["pattern"].startswith("lexicon:"):
        # handled at lexicon stage; skip here
        return new_token, applied, None
    # if char-level regex and scope char/substr, apply regex replacements
    if scope in ("char","substr","token"):
        try:
            pat = rule.get("pattern")
            if pat:
                regex = re.compile(pat, flags=re.IGNORECASE)
                # preconditions:
                preconds = rule.get("preconditions",[])
                context = {"prev_token": prev_token}
                ok = all(eval_precondition(pc, token, context, lexicon) for pc in preconds) if preconds else True
                if ok:
                    # Only perform substitution if an explicit 'replacement' is provided in the rule.
                    replacement = rule.get("replacement")
                    if not isinstance(replacement, str):
                        # No explicit replacement available; skip to avoid inserting prose.
                        return new_token, False, None
                    new_token_candidate, n = regex.subn(replacement, token)
                    if n > 0 and new_token_candidate != token:
                        new_token = new_token_candidate
                        applied = True
                        applied_rule_id = rule.get("rule_id")
                        return new_token, applied, {"rule_id": applied_rule_id, "confidence": rule_conf}
        except re.error:
            return new_token, False, None
    return new_token, False, None

# ---------- Main normalization for a single sentence ----------
def normalize_text(text, rules_data, lexicon, compiled_regexes, pipeline_cfg):
    # optional lowercasing first to align with prior preprocessing
    try:
        do_lower = pipeline_cfg.get("lowercase", True)
    except Exception:
        do_lower = True
    if do_lower and isinstance(text, str):
        text = text.lower()
    # sentence-level safe regex pass
    applied_rules = []
    text = apply_sentence_regexes(text, compiled_regexes, applied_rules)
    
    # Apply sentence-level rules
    for rule in rules_data["rules"]:
        if not rule.get("active", True):
            continue
        if rule.get("scope") == "sentence":
            pattern = rule.get("pattern")
            replacement = rule.get("replacement")
            if pattern and replacement:
                import re
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    new_text = compiled_pattern.sub(replacement, text)
                    if new_text != text:
                        applied_rules.append({
                            "rule": rule["rule_id"], 
                            "from": text, 
                            "to": new_text, 
                            "confidence": rule.get("confidence_default", 0.95)
                        })
                        text = new_text
                except re.error:
                    pass
    
    tokens = simple_tokenize(text)
    out_tokens = []
    prev = None
    
    for i, tok in enumerate(tokens):
        orig_tok = tok
        ctx = build_context(tokens, i)
        
        # 1) lexicon lookup (contextual, multi-sense)
        mapped, map_conf, sense = apply_lexicon_contextual(tok, ctx, lexicon, pipeline_cfg.get("default_confidence_if_missing", 0.75))
        if mapped is not None:
            if map_conf >= pipeline_cfg.get("auto_apply_min_confidence", 0.9):
                if mapped != "":
                    out_tokens.append(mapped)
                applied_rules.append({"rule": "LEXICON", "token": orig_tok, "to": mapped, "confidence": map_conf})
                prev = mapped
                continue
            else:
                applied_rules.append({"rule": "LEXICON_CANDIDATE", "token": orig_tok, "candidate": mapped, "confidence": map_conf})
        
        # 2) deterministic high-priority rules (auto_apply_min_confidence)
        applied_this = False
        for rule in sorted(rules_data["rules"], key=lambda r: -r.get("priority", 50)):
            if not rule.get("active", True):
                continue
            if rule.get("confidence_default", 0.0) < pipeline_cfg.get("auto_apply_min_confidence", 0.9):
                continue
            preconds = rule.get("preconditions", [])
            ok = all(eval_precondition(pc, tok, ctx, lexicon) for pc in preconds) if preconds else True
            if not ok:
                continue
            new_tok, applied, meta = apply_rule_to_token(rule, tok, prev, compiled_regexes, lexicon)
            if applied:
                out_tokens.append(new_tok)
                applied_rules.append({"rule": rule["rule_id"], "from": tok, "to": new_tok, "confidence": meta.get("confidence") if meta else rule.get("confidence_default")})
                prev = new_tok
                applied_this = True
                break
        
        if applied_this:
            continue
            
        # 3) lower-confidence regex patterns (safe ones)
        er = compiled_regexes.get("ELONGATION_COLLAPSE")
        if er:
            newtok = er["pattern"].sub(er["replacement"], tok)
            if newtok != tok:
                out_tokens.append(newtok)
                applied_rules.append({"rule": "REGEX.ELONGATION_COLLAPSE", "from": tok, "to": newtok, "confidence": 0.99})
                prev = newtok
                continue
        
        # default: preserve
        out_tokens.append(tok)
        prev = tok
    
    normalized = " ".join(out_tokens)
    metadata = {"applied_rules": applied_rules}
    return normalized, metadata

# ---------- Demo driver ----------
def demo_with_sample_texts(sample_texts):
    # load all assets
    print("Loading assets...")
    rules = load_and_merge_rules(RULES_FILE, None)
    regexes = compile_regex_patterns(REGEX_FILE)
    lexicon = load_and_merge_lexica(LEXICA_DIR)
    pipeline_cfg = load_json(PIPE_CFG_FILE)
    # augment rules
    rules, changed = augment_rules(rules, pipeline_cfg)
    errs = validate_rules(rules)
    if errs:
        print("Validation errors:", errs)
    # process sample texts
    results = []
    for s in sample_texts:
        normalized, meta = normalize_text(s, rules, lexicon, regexes, pipeline_cfg)
        results.append({"orig": s, "norm": normalized, "meta": meta})
    # save edits log
    with open(pipeline_cfg.get("logging",{}).get("edits_logfile","normalization_edits.jsonl"), "w", encoding="utf8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Demo completed. Results saved to edits log.")
    return results

# ---------- If run as script ----------
if __name__ == "__main__":
    # sample_texts: user-provided sentences (preprocessed_text)
    sample_texts = [
        "puro coffee date yaya ng mga to amputa siguro masyadong oa pagka manifest ko ng starbucks planner ah. di'ba pwedeng gcash date na lang?",
        "di'ba nag open up si anji ky kyle..unlike nung una na sobrang dala ng bugso ng damdamin at sa ayaw nya may masabi si tj.",
        "kasi if it's not legit, the mere mention of st luke's as the testing center, st luke's could have reacted right away, di'ba? but so far, wala namang ganoon. desperada lang ba talaga sila na mapatalsik si bbm.",
        "uy beh akoooo.. hindi player.",
        "beh bkt bkt k pa nagreply.",
        "ikaw-uwu."
    ]
    res = demo_with_sample_texts(sample_texts)
    for r in res:
        print("ORIG:", r["orig"])
        print("NORM:", r["norm"])
        print("APPLIED:", r["meta"]["applied_rules"])
        print("----")






