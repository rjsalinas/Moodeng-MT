#!/usr/bin/env python3
"""
Compare translations between vanilla mBART50 and the fine-tuned baseline LoRA model.

Usage:
  python comparison_translation.py "Filipino sentence here"

Outputs a simple side-by-side comparison.
"""

import sys

from translate_mbart import translate as translate_mbart
from translate_baseline import translate_text as translate_finetuned


def main():
    if len(sys.argv) < 2:
        print("Usage: python comparison_translation.py 'Filipino text here'")
        return

    text = " ".join(sys.argv[1:])

    hyp_mbart = translate_mbart(text)
    hyp_finetuned = translate_finetuned(text)

    print("\n— Input —")
    print(text)
    print("\n— mBART50 (vanilla) —")
    print(hyp_mbart)
    print("\n— Fine-tuned baseline (LoRA) —")
    print(hyp_finetuned)


if __name__ == "__main__":
    main()





