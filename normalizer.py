import json
import os
import re
from datetime import datetime

class FilipinoNormalizer:
    def __init__(self, registry_path, log_dir):
        self.registry_path = registry_path
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'normalization_log.jsonl')
        # Load once
        with open(registry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Active rules sorted by priority (high first) with safe defaults
        raw_rules = [r for r in data.get('rules', []) if r.get('active', True)]
        self.rules = sorted(
            [self._augment_rule_defaults(r) for r in raw_rules],
            key=lambda x: x.get('priority', 0),
            reverse=True
        )
        self.families = data.get('families', {})

    def normalize_text(self, text, context=None):
        """
        text: string to normalize
        context: optional dict like {"sent_id": "...", "token_id": "..."}
        returns (normalized_text, applied_logs)
        """
        normalized_text = text
        applied_logs = []
        
        # Apply text cleaning rules first (highest priority)
        normalized_text, logs = self._apply_text_cleaning_rules(normalized_text, context)
        applied_logs.extend(logs)

        # Apply gibberish and keyboard smashing rules
        normalized_text, logs = self._apply_gibberish_rules(normalized_text, context)
        applied_logs.extend(logs)

        # Apply hashtag and mention removal
        normalized_text, logs = self._apply_social_media_cleaning(normalized_text, context)
        applied_logs.extend(logs)

        # Apply orthographic normalization rules
        for rule in self.rules:
            rule_id = rule.get('rule_id', '')
            pattern = rule.get('pattern', '')
            
            # Apply rule based on pattern
            if pattern == "o↔u":
                normalized_text, logs = self._apply_o_u_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "e↔i":
                normalized_text, logs = self._apply_e_i_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "y↔i":
                normalized_text, logs = self._apply_y_i_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "ng vs n+g":
                normalized_text, logs = self._apply_ng_n_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "ch↔ts":
                normalized_text, logs = self._apply_ch_ts_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "uw↔w":
                normalized_text, logs = self._apply_uw_w_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "redundant_h":
                normalized_text, logs = self._apply_redundant_h_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "(.)\\1+":
                normalized_text, logs = self._apply_duplicate_char_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "shortcut→standard":
                normalized_text, logs = self._apply_slang_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif pattern == "foreign→filipinized":
                normalized_text, logs = self._apply_loan_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif "space→hyphen" in pattern:
                normalized_text, logs = self._apply_dash_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            elif "CVCV CVCV→CVCV-CVCV" in pattern:
                normalized_text, logs = self._apply_reduplication_rule(normalized_text, rule, context)
                applied_logs.extend(logs)
            else:
                # New: apply generic high-confidence regex-based rules (e.g., ORTH_### from rules.json)
                normalized_text, logs = self._apply_generic_regex_rule(normalized_text, rule, context)
                applied_logs.extend(logs)

        # Apply enhanced normalization rules (NEW)
        normalized_text, logs = self._apply_transposition_rules(normalized_text, context)
        applied_logs.extend(logs)
        
        normalized_text, logs = self._apply_token_split_rules(normalized_text, context)
        applied_logs.extend(logs)
        
        normalized_text, logs = self._apply_token_merge_rules(normalized_text, context)
        applied_logs.extend(logs)
        
        normalized_text, logs = self._apply_enhanced_punctuation_rules(normalized_text, context)
        applied_logs.extend(logs)
        
        normalized_text, logs = self._apply_morphology_rules(normalized_text, context)
        applied_logs.extend(logs)

        # Final cleanup and formatting
        normalized_text, logs = self._apply_final_cleanup(normalized_text, context)
        applied_logs.extend(logs)

        # Convert to lowercase
        normalized_text, logs = self._apply_lowercase_conversion(normalized_text, context)
        applied_logs.extend(logs)

        # Add sentence end periods
        normalized_text, logs = self._apply_sentence_end_periods(normalized_text, context)
        applied_logs.extend(logs)

        # Sentence-level summary log
        if applied_logs:
            self._log_sentence(original=text, normalized=normalized_text, applied_logs=applied_logs, context=context)

        return normalized_text, applied_logs

    def _augment_rule_defaults(self, rule):
        """Ensure new schema fields exist with safe defaults so older code doesn't break."""
        r = dict(rule)
        r.setdefault('legacy_ids', [])
        r.setdefault('confidence_default', 0.75)
        r.setdefault('variation_index', 'medium')
        r.setdefault('priority', 50)
        r.setdefault('preconditions', [])
        r.setdefault('scope', 'token')
        r.setdefault('deprecated', False)
        r.setdefault('active', True)
        return r

    def _apply_generic_regex_rule(self, text, rule, context):
        """
        Conservative, generic handler for regex-style rules in rules.json (e.g., ORTH_###).
        - Applies only if confidence_default >= 0.9
        - Skips rules that reference external lexica: pattern startswith 'lexicon:'
        - Uses canonical_choice as replacement if provided; otherwise no-op
        - Does not attempt to evaluate complex preconditions here
        """
        logs = []
        try:
            # Respect confidence threshold (conservative)
            if rule.get('confidence_default', 0.0) < 0.9:
                return text, logs

            pat = rule.get('pattern')
            if not isinstance(pat, str) or not pat:
                return text, logs

            if pat.startswith('lexicon:'):
                # This normalizer doesn't load merged lexica. Skip.
                return text, logs

            replacement = rule.get('canonical_choice')
            if replacement is None:
                return text, logs

            regex = re.compile(pat, flags=re.IGNORECASE)
            new_text, n = regex.subn(replacement, text)
            if n > 0 and new_text != text:
                logs.append(self._mk_edit_log(
                    rule_id=rule.get('rule_id', 'ORTH_GENERIC'),
                    reason='generic_regex_rule',
                    before=text,
                    after=new_text,
                    span=None,
                    context=context,
                    meta={
                        'pattern': pat,
                        'replacement': replacement,
                        'confidence_default': rule.get('confidence_default')
                    }
                ))
                return new_text, logs
            return text, logs
        except re.error:
            # Invalid regex: ignore silently
            return text, logs

    def _apply_text_cleaning_rules(self, text, context):
        """Apply basic text cleaning rules"""
        logs = []
        original_text = text
        
        # Remove URLs more aggressively
        url_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # Full URLs
            r'www\.[^\s]+',  # www. URLs
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',  # Domain patterns
            r'bit\.ly/[a-zA-Z0-9]+',  # Bit.ly URLs
            r'tiny\.url/[a-zA-Z0-9]+',  # Tiny.url URLs
            r'[a-zA-Z0-9]+\.[a-zA-Z]{2,}',  # Simple domain patterns
        ]
        
        for pattern in url_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove truncated URLs and web artifacts
        web_artifacts = [
            r'chron\.com[^\s]*',  # Chron.com patterns
            r'[a-zA-Z0-9]+\.[a-zA-Z]{2,}\.\.\.',  # Truncated URLs with ...
            r'[a-zA-Z0-9]+\.[a-zA-Z]{2,}\.\.\.',  # Truncated URLs with ...
        ]
        
        for pattern in web_artifacts:
            text = re.sub(pattern, '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-printing characters and extended Unicode
        text = re.sub(r'[^\x20-\x7E\xA0-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF\u2C60-\u2C7F\uA720-\uA7FF]', '', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id="TEXT_CLEAN_01",
                reason="enhanced_text_cleaning",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "Enhanced URL removal, whitespace cleanup, non-printing chars, web artifacts"}
            ))
        
        return text, logs

    def _apply_gibberish_rules(self, text, context):
        """Apply gibberish and keyboard smashing detection rules - preserve English text"""
        logs = []
        original_text = text
        
        # Only remove keyboard smashing patterns that are clearly not English words
        keyboard_patterns = [
            r'\b[qwertyuiop]{6,}\b',  # QWERTY keyboard sequences (longer to avoid English words)
            r'\b[asdfghjkl]{6,}\b',   # ASDF row sequences (longer to avoid English words)
            r'\b[zxcvbnm]{6,}\b',     # ZXCV row sequences (longer to avoid English words)
            r'\b[mnbvcxz]{6,}\b',     # Reverse ZXCV sequences (longer to avoid English words)
            r'\b[lkjhgfdsa]{6,}\b',   # Reverse ASDF sequences (longer to avoid English words)
            r'\b[poiuytrewq]{6,}\b',  # Reverse QWERTY sequences (longer to avoid English words)
        ]
        
        for pattern in keyboard_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                # Only remove if it's clearly not an English word
                if not self._looks_like_english_word(match):
                    text = re.sub(rf'\b{match}\b', '', text, flags=re.IGNORECASE)
                    logs.append(self._mk_edit_log(
                        rule_id="GIBBERISH_01",
                        reason="keyboard_smashing_removal",
                        before=original_text,
                        after=text,
                        span=None,
                        context=context,
                        meta={"removed_pattern": match, "type": "keyboard_sequence"}
                    ))
                    original_text = text
        
        # Remove gibberish patterns but be more conservative with English text
        gibberish_patterns = [
            r'\b[bcdfghjklmnpqrstvwxz]{7,}\b',  # Very long consonant clusters (7+ consonants)
            r'\b[aeiou]{6,}\b',                  # Very long vowel clusters (6+ vowels)
        ]
        
        for pattern in gibberish_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                # Only remove if it's clearly not an English word
                if not self._looks_like_english_word(match):
                    text = re.sub(rf'\b{match}\b', '', text, flags=re.IGNORECASE)
                    logs.append(self._mk_edit_log(
                        rule_id="GIBBERISH_02",
                        reason="gibberish_pattern_removal",
                        before=original_text,
                        after=text,
                        span=None,
                        context=context,
                        meta={"removed_pattern": match, "type": "gibberish_pattern"}
                    ))
                    original_text = text
        
        # Remove random character sequences but preserve English words
        random_patterns = [
            r'\b[a-z0-9]{10,}\b',      # Very long alphanumeric sequences (10+ chars)
        ]
        
        for pattern in random_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                # Only remove if it's clearly not an English word
                if not self._looks_like_english_word(match):
                    text = re.sub(rf'\b{match}\b', '', text, flags=re.IGNORECASE)
                    logs.append(self._mk_edit_log(
                        rule_id="GIBBERISH_03",
                        reason="random_sequence_removal",
                        before=original_text,
                        after=text,
                        span=None,
                        context=context,
                        meta={"removed_pattern": match, "type": "random_sequence"}
                    ))
                    original_text = text
        
        return text, logs

    def _apply_social_media_cleaning(self, text, context):
        """Remove hashtags, mentions, emojis, and social media artifacts - preserve English text"""
        logs = []
        original_text = text
        
        # Remove hashtags completely (remove both # and the text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove mentions (@username) completely
        text = re.sub(r'@\w+', '', text)
        
        # Remove emojis and emoticons more aggressively
        # Remove common emoji patterns
        emoji_patterns = [
            r'[\U0001F600-\U0001F64F]',  # Emoticons
            r'[\U0001F300-\U0001F5FF]',  # Misc symbols and pictographs
            r'[\U0001F680-\U0001F6FF]',  # Transport and map symbols
            r'[\U0001F1E0-\U0001F1FF]',  # Regional indicator symbols
            r'[\U00002600-\U000027BF]',  # Misc symbols
            r'[\U0001F900-\U0001F9FF]',  # Supplemental symbols and pictographs
            r'[\U0001FA70-\U0001FAFF]',  # Symbols and pictographs extended-A
        ]
        
        for pattern in emoji_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove common emoticon patterns
        emoticon_patterns = [
            r':\)|:\(|:D|:P|:O|:S|:X|:Z',  # Basic emoticons
            r'😀|😃|😄|😁|😆|😅|😂|🤣|😊|😇|🙂|🙃|😉|😌|😍|🥰|😘|😗|😙|😚|😋|😛|😝|😜|🤪|🤨|🧐|🤓|😎|🤩|🥳|😏|😒|😞|😔|😟|😕|🙁|☹️|😣|😖|😫|😩|🥺|😢|😭|😤|😠|😡|🤬|🤯|😳|🥵|🥶|😱|😨|😰|😥|😓|🤗|🤔|🤭|🤫|🤥|😶|😐|😑|😯|😦|😧|😮|😲|🥱|😴|😪|😵|🤐|🥴|🤢|🤮|🤧|😷|🤒|🤕|🤑|🤠',
            r'🤡|👻|👽|👾|🤖|😺|😸|😹|😻|😼|😽|🙀|😿|😾|🙈|🙉|🙊|💌|💘|💝|💖|💗|💙|💚|❤️|🧡|💛|💜|🖤|💔|❣️|💕|💞|💓|💗|💖|💘|💝|💟|♥️|💎|🔶|🔷|🔸|🔹|🔺|🔻|💠|🔘|🔴|🟠|🟡|🟢|🔵|🟣|⚫|⚪|🟤|🔺|🔻|💠|🔘|🔴|🟠|🟡|🟢|🔵|🟣|⚫|⚪|🟤',
        ]
        
        for pattern in emoticon_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove social media patterns more aggressively
        social_patterns = [
            r'rt\s+@\w+',           # RT @username
            r'via\s+@\w+',          # via @username
            r'cc\s+@\w+',           # cc @username
            r'follow\s+@\w+',       # follow @username
            r'check\s+@\w+',        # check @username
            r'fb\s*\.\s*com',       # fb.com
            r'www\s*\.',            # www.
            r'http[s]?\s*://',      # http:// or https://
            r'bit\s*\.\s*ly',       # bit.ly
            r'tiny\s*\.\s*url',     # tiny.url
        ]
        
        for pattern in social_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove Korean/Japanese/Chinese characters that might be in hashtags
        text = re.sub(r'[\uAC00-\uD7AF\u3040-\u309F\u30A0-\U0001F9FF]', '', text)
        
        # Remove excessive punctuation and symbols but keep basic punctuation
        # Be more aggressive - remove more non-word characters
        text = re.sub(r'[^\w\s\-\'\.\,\!\?\:\;\(\)]', '', text)
        
        # Clean up multiple punctuation (be gentle) - but preserve ending punctuation
        # Only clean up repeated punctuation that's not at the end
        text = re.sub(r'([!?])\1{2,}(?!$)', r'\1', text)  # Only remove 3+ repeated punctuation (not at end)
        text = re.sub(r'([\.])\1{2,}(?!$)', r'\1', text)  # Only remove 3+ repeated dots (not at end)
        text = re.sub(r'([,]){3,}(?!$)', r',', text)      # Only remove 3+ repeated commas (not at end)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id="SOCIAL_CLEAN_01",
                reason="enhanced_social_media_cleaning",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "aggressive hashtag, mention, emoji, and social media artifact removal"}
            ))
        
        return text, logs

    def _apply_final_cleanup(self, text, context):
        """Apply final cleanup and formatting rules - preserve English text"""
        logs = []
        original_text = text
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove leading punctuation (be conservative)
        text = re.sub(r'^[^\w\s]+', '', text)
        
        # Remove only empty parentheses and brackets
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        
        # Clean up multiple spaces around punctuation (be gentle)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        # Ensure proper spacing after sentences (be conservative)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Clean up repeated punctuation marks (ENHANCED) - but preserve ending punctuation
        # Remove multiple periods, exclamation marks, question marks, commas (not at end)
        text = re.sub(r'([.!?])\1{2,}(?!$)', r'\1', text)  # !!! → !, ??? → ?, ... → . (not at end)
        text = re.sub(r'([,;:])\1{2,}(?!$)', r'\1', text)  # ,,, → ,, ;;; → ; (not at end)
        
        # Handle mixed repeated punctuation in the middle (like !!!... → !)
        text = re.sub(r'([!?])\1{2,}[.]{2,}(?!$)', r'\1', text)  # !!!... → ! (not at end)
        text = re.sub(r'[.]{2,}([!?])\1{2,}(?!$)', r'\1', text)  # ...!!! → ! (not at end)
        
        # Clean up ending punctuation - reduce multiple to single, but preserve the type
        if text and text[-1] in '!?':
            # If ending with ! or ?, reduce multiple to single
            text = re.sub(r'([!?])\1+$', r'\1', text)
        elif text and text[-1] == '.':
            # If ending with ., reduce multiple to single
            text = re.sub(r'\.+$', '.', text)
        elif text and text[-1] in ',;:':
            # If ending with ,;:, reduce multiple to single
            text = re.sub(r'([,;:])\1+$', r'\1', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id="FINAL_CLEAN_01",
                reason="final_formatting_cleanup",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "enhanced punctuation cleanup with ending preservation"}
            ))
        
        return text, logs

    def _looks_like_real_word(self, text):
        """Check if a text pattern looks like it could be a real word"""
        # Common Filipino word patterns
        filipino_patterns = [
            r'^[aeiou]',           # Starts with vowel
            r'[aeiou]$',           # Ends with vowel
            r'[aeiou].*[aeiou]',   # Contains at least two vowels
            r'^[bcdfghjklmnpqrstvwxz][aeiou]',  # Consonant-vowel start
            r'[aeiou][bcdfghjklmnpqrstvwxz]$',  # Vowel-consonant end
        ]
        
        # If it matches any Filipino pattern, it might be a real word
        for pattern in filipino_patterns:
            if re.match(pattern, text.lower()):
                return True
        
        # If it's too short, it might be a real word
        if len(text) <= 3:
            return True
        
        # If it contains common Filipino syllables, it might be real
        common_syllables = ['ka', 'sa', 'na', 'ng', 'ang', 'mga', 'si', 'ni', 'ay', 'at', 'nga', 'din', 'rin', 'man', 'lang', 'pa', 'na', 'ba', 'da', 'ga', 'ha', 'la', 'ma', 'pa', 'ra', 'ta', 'wa', 'ya']
        for syllable in common_syllables:
            if syllable in text.lower():
                return True
        
        return False

    def _looks_like_english_word(self, text):
        """Check if a text pattern looks like it could be an English word"""
        # Common English word patterns
        english_patterns = [
            r'^[aeiou]',           # Starts with vowel
            r'[aeiou]$',           # Ends with vowel
            r'[aeiou].*[aeiou]',   # Contains at least two vowels
            r'^[bcdfghjklmnpqrstvwxz][aeiou]',  # Consonant-vowel start
            r'[aeiou][bcdfghjklmnpqrstvwxz]$',  # Vowel-consonant end
        ]
        
        # If it matches any English pattern, it might be a real word
        for pattern in english_patterns:
            if re.match(pattern, text.lower()):
                return True
        
        # If it's too short, it might be a real word
        if len(text) <= 4:
            return True
        
        # If it contains common English syllables, it might be real
        common_english_syllables = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']
        for syllable in common_english_syllables:
            if syllable in text.lower():
                return True
        
        return False

    def _apply_o_u_rule(self, text, rule, context):
        """Apply o↔u alternation rule"""
        logs = []
        original_text = text
        
        # Common o↔u alternations
        replacements = [
            ("kumusta", "kamusta"),
            ("baso", "baso"),  # Keep as is for Filipinized loans
        ]
        
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                logs.append(self._mk_edit_log(
                    rule_id=rule['rule_id'],
                    reason="spanish_borrowing_o_u_variant",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"from_word": old, "to_word": new}
                ))
                original_text = text
        
        return text, logs

    def _apply_e_i_rule(self, text, rule, context):
        """Apply e↔i alternation rule"""
        logs = []
        original_text = text
        
        replacements = [
            ("babae", "babai"),
            ("lalake", "lalaki"),
            ("hangaren", "hangarin"),
            ("tange", "tangi"),
        ]
        
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                logs.append(self._mk_edit_log(
                    rule_id=rule['rule_id'],
                    reason="informal_variant_e_i",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"from_word": old, "to_word": new}
                ))
                original_text = text
        
        return text, logs

    def _apply_y_i_rule(self, text, rule, context):
        """Apply y↔i alternation rule"""
        logs = []
        original_text = text
        
        replacements = [
            ("kolehiala", "kolehiyala"),
        ]
        
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                logs.append(self._mk_edit_log(
                    rule_id=rule['rule_id'],
                    reason="y_i_standardization",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"from_word": old, "to_word": new}
                ))
                original_text = text
        
        return text, logs

    def _apply_ng_n_rule(self, text, rule, context):
        """Apply ng vs n+g rule"""
        logs = []
        original_text = text
        
        # Remove hyphens in ng sequences
        text = re.sub(r'pan-gitna', 'panggitna', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id=rule['rule_id'],
                reason="ng_hyphen_removal",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "pan-gitna → panggitna"}
            ))
        
        return text, logs

    def _apply_ch_ts_rule(self, text, rule, context):
        """Apply ch↔ts rule"""
        logs = []
        original_text = text
        
        replacements = [
            ("chocolate", "tsokolate"),
        ]
        
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                logs.append(self._mk_edit_log(
                    rule_id=rule['rule_id'],
                    reason="ch_ts_filipinization",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"from_word": old, "to_word": new}
                ))
                original_text = text
        
        return text, logs

    def _apply_uw_w_rule(self, text, rule, context):
        """Apply uw↔w rule"""
        logs = []
        original_text = text
        
        # Simplify uw to w in certain contexts
        text = re.sub(r'\bbuwaya\b', 'bwaya', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id=rule['rule_id'],
                reason="uw_w_simplification",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "buwaya → bwaya"}
            ))
        
        return text, logs

    def _apply_redundant_h_rule(self, text, rule, context):
        """Apply redundant h removal rule"""
        logs = []
        original_text = text
        
        # Remove duplicate h
        text = re.sub(r'mahhalaga', 'mahalaga', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id=rule['rule_id'],
                reason="redundant_h_removal",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "mahhalaga → mahalaga"}
            ))
        
        return text, logs

    def _apply_duplicate_char_rule(self, text, rule, context):
        """Apply duplicate character removal rule"""
        logs = []
        original_text = text
        
        # Remove unintended duplicate characters (but preserve intentional ones)
        text = re.sub(r'helllo', 'hello', text)
        text = re.sub(r'sigeee', 'sige', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id=rule['rule_id'],
                reason="duplicate_character_removal",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "duplicate character cleanup"}
            ))
        
        return text, logs

    def _apply_slang_rule(self, text, rule, context):
        """Apply slang normalization rule - map SMS/chat shortcuts to standard Filipino/English"""
        logs = []
        original_text = text
        
        # Comprehensive slang mappings
        slang_mappings = [
            # Filipino shortcuts
            (r'\bq\b', 'ako'),           # q → ako
            (r'\bu\b', 'ikaw'),          # u → ikaw
            (r'\bsya\b', 'siya'),        # sya → siya
            (r'\bnga\b', 'nga'),         # nga → nga (preserve)
            (r'\bngay\b', 'ngayon'),     # ngay → ngayon
            (r'\bngayon\b', 'ngayon'),   # ngayon → ngayon (preserve)
            
            # English shortcuts
            (r'\b2\b', 'to'),            # 2 → to
            (r'\b4\b', 'for'),           # 4 → for
            (r'\b8\b', 'ate'),           # 8 → ate
            (r'\br\b', 'are'),           # r → are
            (r'\by\b', 'why'),           # y → why
            (r'\bgr8\b', 'great'),       # gr8 → great
            (r'\bthx\b', 'thanks'),      # thx → thanks
            (r'\bpls\b', 'please'),      # pls → please
            (r'\bplz\b', 'please'),      # plz → please
            (r'\bomg\b', 'oh my god'),   # omg → oh my god
            (r'\blol\b', 'laugh out loud'), # lol → laugh out loud
            (r'\bbrb\b', 'be right back'), # brb → be right back
            (r'\bafk\b', 'away from keyboard'), # afk → away from keyboard
        ]
        
        for pattern, replacement in slang_mappings:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logs.append(self._mk_edit_log(
                    rule_id=rule['rule_id'],
                    reason="slang_to_standard",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"from_word": pattern, "to_word": replacement, "type": "slang_expansion"}
                ))
                original_text = text
        
        return text, logs

    def _apply_loan_rule(self, text, rule, context):
        """Apply loanword normalization rule"""
        logs = []
        original_text = text
        
        replacements = [
            ("favorite", "paborito"),
            ("window", "bintana"),
        ]
        
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                logs.append(self._mk_edit_log(
                    rule_id=rule['rule_id'],
                    reason="loanword_filipinization",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"from_word": old, "to_word": new}
                ))
                original_text = text
        
        return text, logs

    def _apply_dash_rule(self, text, rule, context):
        """Apply dash normalization rule"""
        logs = []
        original_text = text
        
        # Add hyphens for affix boundaries
        text = re.sub(r'mag aral', 'mag-aral', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id=rule['rule_id'],
                reason="affix_hyphen_addition",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "mag aral → mag-aral"}
            ))
        
        return text, logs

    def _apply_reduplication_rule(self, text, rule, context):
        """Apply reduplication hyphenation rule"""
        logs = []
        original_text = text
        
        # Add hyphens for reduplication
        text = re.sub(r'sama sama', 'sama-sama', text)
        text = re.sub(r'halo halo', 'halo-halo', text)
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id=rule['rule_id'],
                reason="reduplication_hyphenation",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "reduplication hyphenation"}
            ))
        
        return text, logs

    def _apply_transposition_rules(self, text, context):
        """Apply transposition rules for letter-order swaps (typos/metathesis)"""
        logs = []
        original_text = text
        
        # Common Filipino transposition patterns
        transposition_patterns = [
            (r'\balakt\b', 'aklat'),      # alakt → aklat
            (r'\bklat\b', 'aklat'),       # klat → aklat (missing initial vowel)
            (r'\bngay\b', 'ay'),          # ngay → ay (remove ng prefix)
        ]
        
        for pattern, replacement in transposition_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logs.append(self._mk_edit_log(
                    rule_id="TRANSPOSITION_01",
                    reason="letter_order_swap",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"pattern": pattern, "replacement": replacement, "type": "transposition"}
                ))
                original_text = text
        
        return text, logs

    def _apply_token_split_rules(self, text, context):
        """Apply token split rules for missegmentation"""
        logs = []
        original_text = text
        
        # Split incorrectly glued tokens
        split_patterns = [
            (r'\bnakapunta\b', 'naka punta'),      # nakapunta → naka punta
            (r'\bnagprint\b', 'nag print'),        # nagprint → nag print
            (r'\bnagdownload\b', 'nag download'),  # nagdownload → nag download
            (r'\bnagpost\b', 'nag post'),          # nagpost → nag post
            (r'\bnagshare\b', 'nag share'),        # nagshare → nag share
        ]
        
        for pattern, replacement in split_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logs.append(self._mk_edit_log(
                    rule_id="TOKEN_SPLIT_01",
                    reason="token_separation",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"pattern": pattern, "replacement": replacement, "type": "split"}
                ))
                original_text = text
        
        return text, logs

    def _apply_token_merge_rules(self, text, context):
        """Apply token merge rules for wrongly separated tokens"""
        logs = []
        original_text = text
        
        # Merge wrongly separated tokens
        merge_patterns = [
            (r'\bna ka punta\b', 'nakapunta'),      # na ka punta → nakapunta
            (r'\bnag print\b', 'nagprint'),         # nag print → nagprint
            (r'\bna ka download\b', 'nakadownload'), # na ka download → nakadownload
            (r'\bna ka post\b', 'nakapost'),        # na ka post → nakapost
            (r'\bna ka share\b', 'nakashare'),      # na ka share → nakashare
        ]
        
        for pattern, replacement in merge_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logs.append(self._mk_edit_log(
                    rule_id="TOKEN_MERGE_01",
                    reason="token_combination",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"pattern": pattern, "replacement": replacement, "type": "merge"}
                ))
                original_text = text
        
        return text, logs

    def _apply_enhanced_punctuation_rules(self, text, context):
        """Apply enhanced punctuation rules for contractions and apostrophes"""
        logs = []
        original_text = text
        
        # Fix contractions and apostrophes
        punctuation_patterns = [
            (r'\bdi ba\b', "di'ba"),      # di ba → di'ba (contraction)
            (r'\bna nga\b', "na'nga"),    # na nga → na'nga (contraction)
            (r'\bpa nga\b', "pa'nga"),    # pa nga → pa'nga (contraction)
            (r'\bka nga\b', "ka'nga"),    # ka nga → ka'nga (contraction)
            (r'\bko nga\b', "ko'nga"),    # ko nga → ko'nga (contraction)
        ]
        
        for pattern, replacement in punctuation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logs.append(self._mk_edit_log(
                    rule_id="PUNCTUATION_ENH_01",
                    reason="contraction_fix",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"pattern": pattern, "replacement": replacement, "type": "contraction"}
                ))
                original_text = text
        
        return text, logs

    def _apply_morphology_rules(self, text, context):
        """Apply morphology-aware normalization rules"""
        logs = []
        original_text = text
        
        # Morphology patterns for Filipino
        morphology_patterns = [
            # Infix -um- patterns
            (r'\bum\b', 'um'),            # Preserve um infix
            (r'\bum\b', 'um'),            # Preserve um infix
            
            # Reduplication patterns
            (r'\b(\\w+)\\s+\\1\\b', r'\\1-\\1'),  # araw araw → araw-araw
            
            # Affix boundary patterns
            (r'\b(na|ma|pa|ka|sa)\\s+([a-z]+)\\b', r'\\1-\\2'),  # na punta → na-punta
        ]
        
        for pattern, replacement in morphology_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                logs.append(self._mk_edit_log(
                    rule_id="MORPHOLOGY_01",
                    reason="morphology_normalization",
                    before=original_text,
                    after=text,
                    span=None,
                    context=context,
                    meta={"pattern": pattern, "replacement": replacement, "type": "morphology"}
                ))
                original_text = text
        
        return text, logs

    def _apply_lowercase_conversion(self, text, context):
        """Convert all text to lowercase"""
        logs = []
        original_text = text
        
        # Convert to lowercase
        text = text.lower()
        
        if text != original_text:
            logs.append(self._mk_edit_log(
                rule_id="CASE_NORM_01",
                reason="lowercase_conversion",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "convert all text to lowercase"}
            ))
        
        return text, logs

    def _apply_sentence_end_periods(self, text, context):
        """Preserve original ending punctuation or add period if none exists"""
        logs = []
        original_text = text
        
        # Remove any trailing whitespace
        text = text.rstrip(' \t\n\r')
        
        # Check if text already ends with punctuation
        if text and text[-1] in '.,!?;:':
            # Text already has ending punctuation, preserve it
            logs.append(self._mk_edit_log(
                rule_id="SENTENCE_END_01",
                reason="preserve_original_punctuation",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "preserve original ending punctuation", "ending": text[-1]}
            ))
        elif text and not text.endswith('.'):
            # No ending punctuation, add a period
            text = text + '.'
            
            logs.append(self._mk_edit_log(
                rule_id="SENTENCE_END_01",
                reason="add_sentence_end_period",
                before=original_text,
                after=text,
                span=None,
                context=context,
                meta={"pattern": "add period at sentence end"}
            ))
        
        return text, logs

    def _mk_edit_log(self, rule_id, reason, before, after, span, context, meta=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rule_id": rule_id,
            "reason": reason,
            "before": before,
            "after": after,
            "span": span
        }
        if context:
            entry.update(context)
        if meta:
            entry["meta"] = meta
        return entry

    def _log_sentence(self, original, normalized, applied_logs, context):
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "original_text": original,
            "normalized_text": normalized,
            "applied_rules": applied_logs
        }
        if context:
            record.update(context)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Example usage (in a separate script like scripts/run_normalization.py)
if __name__ == "__main__":
    normalizer = FilipinoNormalizer(
        registry_path='rules.json',
        log_dir='logs'
    )
    test_text_1 = "Kumusta ka? Babae ako."
    normalized_text_1, logs_1 = normalizer.normalize_text(test_text_1)
    print(f"Original: {test_text_1}\nNormalized: {normalized_text_1}\nLogs: {logs_1}\n")
    
    test_text_2 = "Lalake siya."
    normalized_text_2, logs_2 = normalizer.normalize_text(test_text_2)
    print(f"Original: {test_text_2}\nNormalized: {normalized_text_2}\nLogs: {logs_2}\n")
