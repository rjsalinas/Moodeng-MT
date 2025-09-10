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

# ---------- Config / file paths ----------
BASE = Path(".")
RULES_FILE = BASE / "rules.json"
REGEX_FILE = BASE / "regex_patterns.json"
LEXICA_DIR = BASE / "lexica"
PIPE_CFG_FILE = BASE / "pipeline_config.json"

# ---------- Helpers to load files ----------
def load_json(p):
    with open(p, "r", encoding="utf8") as f:
        return json.load(f)

def save_json(data, p):
    with open(p, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
    if not Path(lexica_dir).exists():
        return lex
    for p in Path(lexica_dir).glob("*.json"):
        d = load_json(p)
        for token, info in d.items():
            t = token.lower()
            if t in lex:
                # if duplicate, keep higher confidence mapping
                if info.get("confidence",0) > lex[t].get("confidence",0):
                    lex[t] = info
            else:
                lex[t] = info
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
        try:
            compiled[name] = {
                "pattern": re.compile(obj["pattern"], flags=re.IGNORECASE),
                "replacement": obj.get("replacement"),
                "description": obj.get("description", "")
            }
        except re.error as e:
            print(f"Invalid regex for {name}: {e}")
    return compiled

# ---------- Precondition evaluation (simple interpreter) ----------
def eval_precondition(precond_str, token, context, lexicon):
    # Very small "DSL": handle a few patterns
    if not precond_str:
        return True
    s = precond_str.strip()
    if s.startswith("lang in"):
        # check token language using fallback heuristics
        return token_is_fil(token)
    if s == "not_proper_name":
        return not is_proper_name(token)
    if s.startswith("token_in_lexicon"):
        # e.g. token_in_lexicon('slang_lexicon') -> we merged, so just check merged lexicon
        return token.lower() in lexicon
    if s == "morph.is_reduplication == true":
        # expect context = (prev_token, token)
        prev = context.get("prev_token")
        if not prev:
            return False
        return morph_is_reduplication(prev, token)
    # default conservative:
    return False

# ---------- Apply lexicon mappings first ----------
def apply_lexicon(token, lexicon):
    t = token.lower()
    if t in lexicon:
        return lexicon[t]["to"], True, lexicon[t].get("confidence",0.9)
    return token, False, 1.0

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
                    # For now, we just perform a simple substitution to canonical_choice if provided.
                    replacement = rule.get("canonical_choice")
                    if replacement is None:
                        # If no canonical_choice, do not perform blind removal; keep token.
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
    tokens = simple_tokenize(text)
    out_tokens = []
    applied_rules = []
    prev = None
    for i,tok in enumerate(tokens):
        orig_tok = tok
        # 1) lexicon lookup (highest priority)
        mapped, did_map, map_conf = apply_lexicon(tok, lexicon)
        if did_map:
            out_tokens.append(mapped)
            applied_rules.append({"rule":"LEXICON", "token":orig_tok, "to":mapped, "confidence":map_conf})
            prev = mapped
            continue
        # 2) deterministic high-priority rules (auto_apply_min_confidence)
        for rule in sorted(rules_data["rules"], key=lambda r: -r.get("priority",50)):
            if not rule.get("active", True):
                continue
            if rule.get("confidence_default",0.0) < pipeline_cfg.get("auto_apply_min_confidence", 0.9):
                # skip for auto application
                continue
            # Evaluate preconditions
            preconds = rule.get("preconditions",[])
            context = {"prev_token": prev}
            ok = all(eval_precondition(pc, tok, context, lexicon) for pc in preconds) if preconds else True
            if not ok:
                continue
            # Apply substitution via regex or pattern
            new_tok, applied, meta = apply_rule_to_token(rule, tok, prev, compiled_regexes, lexicon)
            if applied:
                out_tokens.append(new_tok)
                applied_rules.append({"rule": rule["rule_id"], "from": tok, "to": new_tok, "confidence": meta.get("confidence") if meta else rule.get("confidence_default")})
                prev = new_tok
                break
        else:
            # 3) lower-confidence rules or regex patterns: check the named regexes
            # apply ELONGATION_COLLAPSE always (collapse repeated chars)
            er = compiled_regexes.get("ELONGATION_COLLAPSE")
            if er:
                newtok = er["pattern"].sub(er["replacement"], tok)
                if newtok != tok:
                    out_tokens.append(newtok)
                    applied_rules.append({"rule":"REGEX.ELONGATION_COLLAPSE","from":tok,"to":newtok,"confidence":0.99})
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
    rules = load_json(RULES_FILE)
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






