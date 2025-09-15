#!/usr/bin/env python3
"""
Enhanced Filipino Text Preprocessing with CalamanCy Integration

This module enhances the existing preprocessing pipeline with:
- Tagalog-specific linguistic complexity calculation
- Advanced Filipino morphological augmentation
- Quality validation using Tagalog grammar rules
- Filipino-aware tokenization with CalamanCy
"""

import spacy
import pandas as pd
import re
from typing import List, Tuple, Dict, Any
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import CalamanCy
try:
    import calamancy
    CALAMANCY_AVAILABLE = True
    logger.info("✅ CalamanCy imported successfully")
except ImportError:
    CALAMANCY_AVAILABLE = False
    logger.warning("⚠️  CalamanCy not available, falling back to basic spaCy")

class EnhancedFilipinoPreprocessor:
    """Enhanced preprocessing pipeline using CalamanCy for Tagalog-specific features."""
    
    def __init__(self, use_calamancy: bool = True):
        """Initialize the preprocessor with CalamanCy or spaCy fallback."""
        self.nlp = None
        self.use_calamancy = use_calamancy and CALAMANCY_AVAILABLE
        
        if self.use_calamancy:
            self._initialize_calamancy()
        else:
            self._initialize_spacy_fallback()
    
    def _initialize_calamancy(self):
        """Initialize CalamanCy with Tagalog-specific models."""
        # CalamanCy available models from the error message
        calamancy_models = [
            'tl_calamancy_md-0.2.0',  # Medium model, latest version
            'tl_calamancy_lg-0.2.0',  # Large model, latest version
            'tl_calamancy_trf-0.2.0', # Transformer model, latest version
            'tl_calamancy_md-0.1.0',  # Medium model, older version
            'tl_calamancy_lg-0.1.0',  # Large model, older version
            'tl_calamancy_trf-0.1.0'  # Transformer model, older version
        ]
        
        for model_name in calamancy_models:
            try:
                self.nlp = calamancy.load(model_name)
                logger.info(f"✅ CalamanCy Tagalog model '{model_name}' loaded successfully")
                
                # Add sentence boundary detection if not present
                if 'sentencizer' not in self.nlp.pipe_names:
                    self.nlp.add_pipe('sentencizer')
                    logger.info("✅ Added sentence boundary detection")
                
                return  # Successfully loaded, exit
                
            except Exception as e:
                logger.warning(f"⚠️  CalamanCy model '{model_name}' failed to load: {e}")
                continue
        
        # If all CalamanCy models failed
        logger.warning("⚠️  All CalamanCy models failed to load")
        logger.info("🔄 Falling back to spaCy multilingual model")
        self.use_calamancy = False
        self._initialize_spacy_fallback()
    
    def _initialize_spacy_fallback(self):
        """Initialize spaCy as fallback."""
        model_options = [
            'xx_ent_wiki_sm',  # Multilingual entity model
            'xx_sent_ud_sm',   # Multilingual sentence model
            'en_core_web_sm'   # English model as fallback
        ]
        
        for model in model_options:
            try:
                self.nlp = spacy.load(model)
                logger.info(f"✅ spaCy model '{model}' loaded as fallback")
                
                # Add sentence boundary detection
                if 'sentencizer' not in self.nlp.pipe_names:
                    self.nlp.add_pipe('sentencizer')
                    logger.info("✅ Added sentence boundary detection")
                
                break
            except OSError:
                logger.warning(f"⚠️  Model '{model}' not found, trying next option...")
                continue
        
        if self.nlp is None:
            logger.error("❌ No spaCy models available")
            self.nlp = None
    
    def enhanced_complexity_calculation(self, text: str) -> Dict[str, Any]:
        """Calculate Tagalog-specific linguistic complexity using CalamanCy."""
        if not self.nlp or not text or pd.isna(text):
            return {
                'total_score': 0,
                'word_count': 0,
                'sentence_count': 0,
                'pos_complexity': 0,
                'dependency_complexity': 0,
                'morphological_complexity': 0,
                'entity_complexity': 0,
                'tagalog_specific_score': 0
            }
        
        try:
            doc = self.nlp(text)
            
            # Basic metrics
            word_count = len([token for token in doc if not token.is_space])
            
            # Handle sentence count
            try:
                sentence_count = len(list(doc.sents))
            except Exception:
                sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
            
            # Tagalog-specific complexity features
            tagalog_specific_score = 0
            
            if self.use_calamancy:
                # Use CalamanCy's Tagalog-specific features
                tagalog_specific_score = self._calculate_tagalog_complexity(doc)
            else:
                # Fallback to generic spaCy analysis
                tagalog_specific_score = self._calculate_generic_complexity(doc)
            
            # Calculate total complexity score
            total_score = (
                word_count * 0.3 +
                sentence_count * 2.0 +
                tagalog_specific_score * 2.5  # Higher weight for Tagalog features
            )
            
            return {
                'total_score': round(total_score, 2),
                'word_count': word_count,
                'sentence_count': sentence_count,
                'tagalog_specific_score': tagalog_specific_score,
                'pos_complexity': tagalog_specific_score // 2,  # Simplified for compatibility
                'dependency_complexity': 0,
                'morphological_complexity': 0,
                'entity_complexity': 0
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Complexity calculation failed for text: {e}")
            return {'total_score': len(str(text)), 'word_count': len(str(text).split())}
    
    def _calculate_tagalog_complexity(self, doc) -> float:
        """Calculate complexity using CalamanCy's Tagalog-specific features."""
        score = 0
        
        try:
            # Tagalog verb complexity (affix patterns)
            for token in doc:
                if token.pos_ == 'VERB':
                    # Check for Tagalog verb affixes
                    if any(affix in token.text for affix in ['mag-', 'nag-', 'naka-', 'nakapag-', 'ma-', 'na-']):
                        score += 2.0
                    # Check for reduplication
                    if self._has_reduplication(token.text):
                        score += 1.5
                    # Check for complex verb forms
                    if len(token.text) > 8:  # Long verbs often indicate complexity
                        score += 1.0
            
            # Tagalog particle complexity
            tagalog_particles = ['ng', 'sa', 'ang', 'ay', 'na', 'pa', 'din', 'rin', 'ko', 'mo', 'ni', 'nila']
            particle_count = sum(1 for token in doc if token.text.lower() in tagalog_particles)
            score += particle_count * 0.5
            
            # Tagalog-specific sentence patterns
            if self._has_tagalog_sentence_pattern(doc):
                score += 3.0
            
            # Entity complexity (Filipino names, places)
            entity_score = len(doc.ents) * 2.0
            score += entity_score
            
        except Exception as e:
            logger.warning(f"⚠️  Tagalog complexity calculation failed: {e}")
            score = 0
        
        return score
    
    def _calculate_generic_complexity(self, doc) -> float:
        """Calculate complexity using generic spaCy features."""
        score = 0
        
        try:
            # POS-based complexity
            pos_counts = {}
            for token in doc:
                if token.pos_ not in pos_counts:
                    pos_counts[token.pos_] = 0
                pos_counts[token.pos_] += 1
            
            pos_complexity = sum(pos_counts.get(pos, 0) for pos in ['VERB', 'ADJ', 'ADV', 'NOUN'])
            score += pos_complexity * 0.5
            
            # Dependency complexity
            dependency_complexity = len([token.dep_ for token in doc 
                                      if token.dep_ in ['nsubj', 'dobj', 'iobj', 'compound', 'amod']])
            score += dependency_complexity * 0.3
            
        except Exception as e:
            logger.warning(f"⚠️  Generic complexity calculation failed: {e}")
            score = 0
        
        return score
    
    def _has_reduplication(self, text: str) -> bool:
        """Check if Tagalog text has reduplication patterns."""
        # Common Tagalog reduplication patterns
        reduplication_patterns = [
            r'(\w{1,3})\1',  # CVCV pattern
            r'(\w{2})\1',     # CV pattern
        ]
        
        for pattern in reduplication_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_tagalog_sentence_pattern(self, doc) -> bool:
        """Check for Tagalog-specific sentence patterns."""
        try:
            # Check for VSO (Verb-Subject-Object) pattern common in Tagalog
            tokens = [token for token in doc if not token.is_space]
            if len(tokens) >= 3:
                # Look for verb at beginning followed by subject/object
                if (tokens[0].pos_ == 'VERB' and 
                    any(t.pos_ in ['NOUN', 'PROPN'] for t in tokens[1:3])):
                    return True
                
                # Check for "ay" pattern (Tagalog copula)
                if any(token.text.lower() == 'ay' for token in tokens):
                    return True
            
            return False
        except Exception:
            return False
    
    def filipino_morphological_augmentation(self, text: str) -> List[str]:
        """Generate Tagalog-specific morphological variations."""
        if not self.nlp or not text:
            return [text]
        
        augmented_texts = [text]
        
        try:
            doc = self.nlp(text)
            
            if self.use_calamancy:
                # Use CalamanCy's Tagalog-specific augmentation
                augmented_texts.extend(self._tagalog_morphological_variations(doc, text))
            else:
                # Fallback to basic augmentation
                augmented_texts.extend(self._basic_morphological_variations(doc, text))
            
            logger.info(f"✅ Generated {len(augmented_texts)} morphological variations")
            return augmented_texts
            
        except Exception as e:
            logger.warning(f"⚠️  Morphological augmentation failed: {e}")
            return [text]
    
    def _tagalog_morphological_variations(self, doc, text: str) -> List[str]:
        """Generate Tagalog-specific morphological variations."""
        variations = []
        
        try:
            for token in doc:
                if token.pos_ == 'VERB':
                    # Tagalog verb affix variations
                    variations.extend(self._generate_tagalog_verb_forms(token.text, text))
                
                elif token.pos_ == 'NOUN':
                    # Tagalog noun variations
                    variations.extend(self._generate_tagalog_noun_forms(token.text, text))
            
        except Exception as e:
            logger.warning(f"⚠️  Tagalog morphological variations failed: {e}")
        
        return variations
    
    def _generate_tagalog_verb_forms(self, verb: str, text: str) -> List[str]:
        """Generate Tagalog verb form variations."""
        variations = []
        
        # Tagalog verb affix patterns
        affix_patterns = {
            'mag-': ['nag-', 'nag-', 'nakapag-'],
            'nag-': ['mag-', 'nag-', 'nakapag-'],
            'ma-': ['na-', 'ma-', 'naka-'],
            'na-': ['ma-', 'na-', 'naka-']
        }
        
        for prefix, alternatives in affix_patterns.items():
            if verb.startswith(prefix):
                for alt in alternatives:
                    if alt != prefix:
                        new_verb = verb.replace(prefix, alt, 1)
                        new_text = text.replace(verb, new_verb)
                        if new_text != text and new_text not in variations:
                            variations.append(new_text)
                break
        
        return variations
    
    def _generate_tagalog_noun_forms(self, noun: str, text: str) -> List[str]:
        """Generate Tagalog noun form variations."""
        variations = []
        
        # Tagalog pluralization patterns
        if noun.endswith('ng'):
            # Try removing 'ng' suffix
            singular = noun[:-2]
            if singular and singular != noun:
                new_text = text.replace(noun, singular)
                if new_text != text:
                    variations.append(new_text)
        
        return variations
    
    def _basic_morphological_variations(self, doc, text: str) -> List[str]:
        """Generate basic morphological variations as fallback."""
        variations = []
        
        try:
            for token in doc:
                if token.pos_ == 'VERB':
                    # Simple verb variations
                    if 'Tense=Pres' in token.morph:
                        past_form = self._get_past_form(token.text)
                        if past_form and past_form != token.text:
                            new_text = text.replace(token.text, past_form)
                            if new_text not in variations:
                                variations.append(new_text)
        except Exception:
            pass
        
        return variations
    
    def _get_past_form(self, verb: str) -> str:
        """Get past tense form of Filipino verb."""
        # Common Filipino verb patterns
        past_patterns = {
            'nag': 'nag',  # Already past
            'nag-': 'nag-',
            'naka': 'naka',
            'nakapag': 'nakapag',
            'nakapag-': 'nakapag-'
        }
        
        for pattern, past_form in past_patterns.items():
            if verb.startswith(pattern):
                return verb  # Already in past form
        
        # Simple past form (add 'nag-' prefix for some verbs)
        if verb.startswith('mag'):
            return verb.replace('mag', 'nag', 1)
        
        return verb
    
    def validate_filipino_quality(self, src: str, tgt: str) -> Dict[str, Any]:
        """Validate the quality of Filipino-English translation pairs using Tagalog-specific rules."""
        if not self.nlp:
            return {'is_valid': True, 'score': 0.8, 'issues': []}
        
        try:
            src_doc = self.nlp(src)
            tgt_doc = self.nlp(tgt)
            
            issues = []
            score = 1.0
            
            if self.use_calamancy:
                # Use CalamanCy's Tagalog-specific validation
                score, issues = self._tagalog_quality_validation(src_doc, tgt_doc, score, issues)
            else:
                # Fallback to generic validation
                score, issues = self._generic_quality_validation(src_doc, tgt_doc, score, issues)
            
            # Ensure score doesn't go below 0
            score = max(0.0, score)
            
            # Be more lenient with validation threshold
            is_valid = score >= 0.3
            
            return {
                'is_valid': is_valid,
                'score': round(score, 2),
                'issues': issues,
                'uses_calamancy': self.use_calamancy
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Quality validation failed: {e}")
            return {'is_valid': True, 'score': 0.7, 'issues': ['Validation error']}
    
    def _tagalog_quality_validation(self, src_doc, tgt_doc, score: float, issues: List[str]) -> Tuple[float, List[str]]:
        """Tagalog-specific quality validation using CalamanCy."""
        try:
            # Check Tagalog-specific structure
            has_tagalog_verb = any(token.pos_ == 'VERB' for token in src_doc)
            has_tagalog_particles = any(token.text.lower() in ['ng', 'sa', 'ang', 'ay'] for token in src_doc)
            
            if not has_tagalog_verb:
                issues.append("Missing Tagalog verb")
                score -= 0.2
            
            if not has_tagalog_particles:
                issues.append("Missing Tagalog particles")
                score -= 0.1
            
            # Check for Tagalog sentence patterns
            if not self._has_tagalog_sentence_pattern(src_doc):
                issues.append("Non-standard Tagalog sentence pattern")
                score -= 0.1
            
        except Exception as e:
            logger.warning(f"⚠️  Tagalog validation failed: {e}")
        
        return score, issues
    
    def _generic_quality_validation(self, src_doc, tgt_doc, score: float, issues: List[str]) -> Tuple[float, List[str]]:
        """Generic quality validation as fallback."""
        try:
            # Basic structure checks
            has_verb = any(token.pos_ == 'VERB' for token in src_doc)
            if not has_verb:
                issues.append("Missing verb in Filipino text")
                score -= 0.2
            
            # Check for common Filipino words
            filipino_indicators = ['ng', 'sa', 'ang', 'ay', 'na', 'pa', 'din', 'rin']
            has_filipino_words = any(token.text.lower() in filipino_indicators for token in src_doc)
            if not has_filipino_words:
                issues.append("No common Filipino words detected")
                score -= 0.05
            
        except Exception as e:
            logger.warning(f"⚠️  Generic validation failed: {e}")
        
        return score, issues
    
    def filipino_aware_tokenization(self, text: str) -> str:
        """Apply Tagalog-aware tokenization using CalamanCy."""
        if not self.nlp or not text:
            return text
        
        try:
            doc = self.nlp(text)
            
            if self.use_calamancy:
                # Use CalamanCy's Tagalog-specific tokenization
                return self._tagalog_tokenization(doc)
            else:
                # Fallback to basic tokenization
                return self._basic_tokenization(doc)
            
        except Exception as e:
            logger.warning(f"⚠️  Tokenization failed: {e}")
            return text
    
    def _tagalog_tokenization(self, doc) -> str:
        """Tagalog-specific tokenization preserving Filipino word boundaries."""
        tokens = []
        
        for token in doc:
            if token.like_num or token.is_alpha:
                tokens.append(token.text)
            else:
                # Handle Tagalog-specific contractions
                if "'" in token.text:
                    # Common Tagalog contractions
                    contractions = {
                        "di'ba": "di ba",
                        "kasi'ng": "kasi ng",
                        "sa'yo": "sa iyo",
                        "ko'ng": "ko ng",
                        "mo'ng": "mo ng",
                        "ni'ng": "ni ng"
                    }
                    
                    if token.text in contractions:
                        tokens.append(contractions[token.text])
                    else:
                        # Split on apostrophe
                        parts = token.text.split("'")
                        tokens.extend(parts)
                else:
                    tokens.append(token.text)
        
        return " ".join(tokens)
    
    def _basic_tokenization(self, doc) -> str:
        """Basic tokenization as fallback."""
        return " ".join([token.text for token in doc if not token.is_space])
    
    def enhance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all Tagalog-specific enhancements to the dataset."""
        if df.empty:
            return df
        
        logger.info("🚀 Starting CalamanCy-enhanced preprocessing...")
        logger.info(f"🔧 Using CalamanCy: {self.use_calamancy}")
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Apply Filipino-aware tokenization
        logger.info("🔧 Applying Tagalog-aware tokenization...")
        df['src_enhanced'] = df['src'].apply(self.filipino_aware_tokenization)
        df['tgt_enhanced'] = df['tgt'].apply(self.filipino_aware_tokenization)
        
        # Calculate enhanced complexity
        logger.info("📊 Calculating Tagalog-specific complexity scores...")
        complexity_results = df['src_enhanced'].apply(self.enhanced_complexity_calculation)
        
        # Extract complexity scores
        df['complexity_score'] = complexity_results.apply(lambda x: x.get('total_score', 0))
        df['word_count'] = complexity_results.apply(lambda x: x.get('word_count', 0))
        df['sentence_count'] = complexity_results.apply(lambda x: x.get('sentence_count', 0))
        df['tagalog_complexity'] = complexity_results.apply(lambda x: x.get('tagalog_specific_score', 0))
        
        # Apply quality validation
        logger.info("✅ Validating translation quality with Tagalog rules...")
        quality_results = df.apply(lambda row: self.validate_filipino_quality(
            row['src_enhanced'], row['tgt_enhanced']
        ), axis=1)
        
        df['quality_score'] = quality_results.apply(lambda x: x.get('score', 0))
        df['quality_valid'] = quality_results.apply(lambda x: x.get('is_valid', True))
        df['quality_issues'] = quality_results.apply(lambda x: x.get('issues', []))
        df['uses_calamancy'] = quality_results.apply(lambda x: x.get('uses_calamancy', False))
        
        # Ensure uses_calamancy column exists and has proper values
        if 'uses_calamancy' not in df.columns:
            df['uses_calamancy'] = self.use_calamancy
        
        # Filter by quality
        initial_count = len(df)
        df = df[df['quality_valid'] == True]
        filtered_count = len(df)
        
        logger.info(f"✅ Quality filtering: {initial_count} → {filtered_count} samples")
        
        # Apply morphological augmentation
        logger.info("🔄 Applying Tagalog morphological augmentation...")
        augmented_rows = []
        
        for _, row in df.iterrows():
            # Ensure all required columns exist
            complexity_score = row.get('complexity_score', 0)
            quality_score = row.get('quality_score', 0.8)
            
            # Original pair
            augmented_rows.append({
                'src': row['src_enhanced'],
                'tgt': row['tgt_enhanced'],
                'complexity_score': complexity_score,
                'quality_score': quality_score,
                'tagalog_complexity': row.get('tagalog_complexity', 0),
                'is_augmented': False
            })
            
            # Generate variations for high-quality samples
            if quality_score >= 0.8:
                variations = self.filipino_morphological_augmentation(row['src_enhanced'])
                
                for variation in variations[1:]:  # Skip original
                    if variation != row['src_enhanced']:
                        # Calculate complexity for variation
                        var_complexity = self.enhanced_complexity_calculation(variation)
                        var_score = var_complexity.get('total_score', complexity_score)
                        var_tagalog_score = var_complexity.get('tagalog_specific_score', 0)
                        
                        augmented_rows.append({
                            'src': variation,
                            'tgt': row['tgt_enhanced'],
                            'complexity_score': var_score,
                            'quality_score': quality_score,
                            'tagalog_complexity': var_tagalog_score,
                            'is_augmented': True
                        })
        
        # Create enhanced dataset
        enhanced_df = pd.DataFrame(augmented_rows)
        
        # Ensure all required columns exist
        if 'complexity_score' not in enhanced_df.columns:
            enhanced_df['complexity_score'] = 0
        if 'quality_score' not in enhanced_df.columns:
            enhanced_df['quality_score'] = 0.8
        if 'tagalog_complexity' not in enhanced_df.columns:
            enhanced_df['tagalog_complexity'] = 0
        if 'uses_calamancy' not in enhanced_df.columns:
            enhanced_df['uses_calamancy'] = self.use_calamancy
        
        # Sort by Tagalog complexity for curriculum learning
        enhanced_df = enhanced_df.sort_values('tagalog_complexity').reset_index(drop=True)
        
        logger.info(f"✅ CalamanCy-enhanced preprocessing completed: {len(enhanced_df)} total samples")
        logger.info(f"📊 Tagalog complexity range: {enhanced_df['tagalog_complexity'].min():.1f} - {enhanced_df['tagalog_complexity'].max():.1f}")
        logger.info(f"📝 Average quality score: {enhanced_df['quality_score'].mean():.2f}")
        logger.info(f"🎯 CalamanCy usage: {enhanced_df['uses_calamancy'].sum()}/{len(enhanced_df)} samples")
        
        return enhanced_df

# Convenience function for easy integration
def enhance_filipino_dataset(df: pd.DataFrame, use_calamancy: bool = True) -> pd.DataFrame:
    """Convenience function to enhance a Filipino dataset with CalamanCy."""
    preprocessor = EnhancedFilipinoPreprocessor(use_calamancy)
    return preprocessor.enhance_dataset(df)
