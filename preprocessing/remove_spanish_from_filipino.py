import pandas as pd
import re
import os
from datetime import datetime

def create_spanish_detection_patterns():
    """
    Create comprehensive Spanish word and pattern detection
    """
    # Common Spanish words that indicate Spanish language
    spanish_words = {
        # Articles and pronouns
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        'yo', 'tú', 'él', 'ella', 'usted', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas', 'ustedes',
        'me', 'te', 'le', 'nos', 'os', 'les', 'se',
        'mi', 'tu', 'su', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'suyo', 'suya', 'suyos', 'suyas',
        
        # Common verbs (present tense)
        'es', 'son', 'está', 'están', 'tiene', 'tienen', 'hace', 'hacen', 'dice', 'dicen', 'va', 'van', 'viene', 'vienen',
        'puede', 'pueden', 'debe', 'deben', 'quiere', 'quieren', 'sabe', 'saben', 'conoce', 'conocen',
        'habla', 'hablan', 'escribe', 'escriben', 'lee', 'leen', 'oye', 'oyen', 've', 'ven',
        
        # Common Spanish words
        'que', 'para', 'por', 'con', 'sin', 'sobre', 'entre', 'detrás', 'delante', 'encima', 'debajo',
        'aquí', 'allí', 'ahí', 'allá', 'cerca', 'lejos', 'dentro', 'fuera', 'arriba', 'abajo',
        'antes', 'después', 'ahora', 'entonces', 'siempre', 'nunca', 'jamás', 'también', 'tampoco',
        'muy', 'más', 'menos', 'poco', 'mucho', 'demasiado', 'bastante', 'casi', 'apenas',
        'bien', 'mal', 'bueno', 'buena', 'buenos', 'buenas', 'malo', 'mala', 'malos', 'malas',
        'grande', 'pequeño', 'pequeña', 'alto', 'alta', 'bajo', 'baja', 'nuevo', 'nueva', 'viejo', 'vieja',
        'bonito', 'bonita', 'feo', 'fea', 'hermoso', 'hermosa', 'lindo', 'linda',
        
        # Days, months, time
        'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo',
        'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
        'hoy', 'ayer', 'mañana', 'tarde', 'noche', 'mañana', 'mediodía', 'medianoche',
        
        # Common phrases
        'buenos días', 'buenas tardes', 'buenas noches', 'por favor', 'gracias', 'de nada', 'perdón', 'lo siento',
        '¿cómo estás?', '¿qué tal?', '¿cómo te va?', 'muy bien', 'muy mal', 'así así',
        'no sé', 'no entiendo', 'no comprendo', '¿puedes ayudarme?', 'claro que sí', 'por supuesto',
        
        # Internet/Social media Spanish
        'jaja', 'jajaja', 'jajajaja', 'lol', 'omg', 'wtf', 'fuck', 'shit', 'damn',
        'amigo', 'amiga', 'hermano', 'hermana', 'mamá', 'papá', 'familia', 'casa', 'trabajo', 'escuela',
        'amor', 'vida', 'mundo', 'tiempo', 'persona', 'gente', 'hombre', 'mujer', 'niño', 'niña',
        'problema', 'solución', 'idea', 'pensamiento', 'sentimiento', 'emoción', 'razón', 'verdad', 'mentira',
        
        # Spanish-specific patterns
        'ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü',  # Spanish diacritics
        '¿', '¡',  # Spanish punctuation
    }
    
    # Spanish verb conjugations (common patterns)
    spanish_verb_patterns = [
        r'\b\w+ar\b',  # -ar verbs
        r'\b\w+er\b',  # -er verbs  
        r'\b\w+ir\b',  # -ir verbs
        r'\b\w+ando\b',  # -ando (gerund)
        r'\b\w+iendo\b',  # -iendo (gerund)
        r'\b\w+ado\b',  # -ado (past participle)
        r'\b\w+ido\b',  # -ido (past participle)
    ]
    
    return spanish_words, spanish_verb_patterns

def detect_spanish_content(text, spanish_words, spanish_verb_patterns):
    """
    Detect if text contains significant Spanish content
    Returns: (is_spanish, spanish_words_found, confidence_score)
    """
    if not text or pd.isna(text):
        return False, [], 0.0
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return False, [], 0.0
    
    # Count Spanish words
    spanish_words_found = []
    spanish_word_count = 0
    
    for word in words:
        if word in spanish_words:
            spanish_words_found.append(word)
            spanish_word_count += 1
    
    # Check for Spanish verb patterns
    spanish_verb_count = 0
    for pattern in spanish_verb_patterns:
        matches = re.findall(pattern, text_lower)
        spanish_verb_count += len(matches)
    
    # Check for Spanish diacritics and punctuation
    spanish_chars = len(re.findall(r'[ñáéíóúü¿¡]', text))
    
    # Calculate confidence score
    total_indicators = spanish_word_count + spanish_verb_count + spanish_chars
    confidence_score = total_indicators / len(words) if len(words) > 0 else 0.0
    
    # Consider it Spanish if:
    # 1. High confidence (>0.3) OR
    # 2. Multiple Spanish indicators (>5) OR  
    # 3. Contains Spanish diacritics
    is_spanish = (confidence_score > 0.3 or 
                  total_indicators > 5 or 
                  spanish_chars > 0)
    
    return is_spanish, spanish_words_found, confidence_score

def filter_spanish_from_filipino_csv(input_csv, output_csv, log_file):
    """
    Remove Spanish tweets from Filipino-only CSV file
    """
    print(f"🔍 Starting Spanish content filtering...")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    print(f"Log file: {log_file}")
    
    # Load Spanish detection patterns
    spanish_words, spanish_verb_patterns = create_spanish_detection_patterns()
    print(f"✓ Loaded {len(spanish_words)} Spanish words and {len(spanish_verb_patterns)} verb patterns")
    
    # Read the input CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"✓ Loaded {len(df)} tweets from {input_csv}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        return
    
    # Check required columns
    if 'text' not in df.columns:
        print(f"✗ 'text' column not found. Available columns: {list(df.columns)}")
        return
    
    # Initialize tracking
    total_tweets = len(df)
    spanish_tweets = 0
    filipino_tweets = 0
    removed_tweets = []
    
    # Process each tweet
    print("\n🔍 Analyzing tweets for Spanish content...")
    
    for index, row in df.iterrows():
        text = row['text']
        tweet_id = row.get('id', index)
        
        # Detect Spanish content
        is_spanish, spanish_words_found, confidence = detect_spanish_content(text, spanish_words, spanish_verb_patterns)
        
        if is_spanish:
            spanish_tweets += 1
            removed_tweets.append({
                'tweet_id': tweet_id,
                'text': text,
                'spanish_words': spanish_words_found,
                'confidence': confidence,
                'reason': 'Spanish content detected'
            })
            
            # Progress indicator
            if spanish_tweets % 10 == 0:
                print(f"  Found {spanish_tweets} Spanish tweets...")
        else:
            filipino_tweets += 1
    
    # Filter out Spanish tweets
    df_filtered = df[~df.index.isin([i for i, row in df.iterrows() 
                                    if detect_spanish_content(row['text'], spanish_words, spanish_verb_patterns)[0]])]
    
    # Save filtered data
    try:
        df_filtered.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✓ Saved filtered data to: {output_csv}")
    except Exception as e:
        print(f"✗ Error saving output file: {e}")
        return
    
    # Save removal log
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Spanish Tweet Removal Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {input_csv}\n")
            f.write(f"Output file: {output_csv}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"  Total tweets: {total_tweets}\n")
            f.write(f"  Spanish tweets removed: {spanish_tweets}\n")
            f.write(f"  Filipino tweets kept: {filipino_tweets}\n")
            f.write(f"  Removal rate: {(spanish_tweets/total_tweets)*100:.1f}%\n\n")
            
            f.write("Removed Tweets:\n")
            f.write("-" * 30 + "\n")
            for tweet in removed_tweets:
                f.write(f"ID: {tweet['tweet_id']}\n")
                f.write(f"Text: {tweet['text']}\n")
                f.write(f"Spanish words: {', '.join(tweet['spanish_words'])}\n")
                f.write(f"Confidence: {tweet['confidence']:.3f}\n")
                f.write(f"Reason: {tweet['reason']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"✓ Saved removal log to: {log_file}")
    except Exception as e:
        print(f"⚠ Warning: Could not save log file: {e}")
    
    # Print summary
    print(f"\n📊 Filtering Results:")
    print(f"  Total tweets processed: {total_tweets}")
    print(f"  Spanish tweets removed: {spanish_tweets}")
    print(f"  Filipino tweets kept: {filipino_tweets}")
    print(f"  Removal rate: {(spanish_tweets/total_tweets)*100:.1f}%")
    
    if removed_tweets:
        print(f"\n🔍 Sample Spanish tweets removed:")
        for i, tweet in enumerate(removed_tweets[:3]):
            print(f"  {i+1}. ID {tweet['tweet_id']}: {tweet['text'][:100]}...")
            print(f"     Spanish words: {', '.join(tweet['spanish_words'][:5])}")
    
    print(f"\n🎉 Spanish filtering complete!")
    print(f"Check the output file: {output_csv}")
    print(f"Check the removal log: {log_file}")

if __name__ == "__main__":
    # File paths
    input_file = "tweets_id_filipino_text_only.csv"
    output_file = "tweets_id_filipino_text_only_no_spanish.csv"
    log_file = "spanish_removal_log.txt"
    
    # Run the filtering
    filter_spanish_from_filipino_csv(input_file, output_file, log_file)
