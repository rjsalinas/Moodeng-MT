import os
from datetime import datetime

os.makedirs('training_logs', exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
path = os.path.join('training_logs', f'training_{ts}.log')

lines = [
    f"{datetime.now().isoformat()} INFO train_loss=2.345678\n",
    f"{datetime.now().isoformat()} INFO val_loss=2.123456\n",
    f"{datetime.now().isoformat()} INFO bleu=0.123456\n",
    f"{datetime.now().isoformat()} INFO best_val_loss=2.123456\n",
    f"{datetime.now().isoformat()} INFO best_bleu=0.123456\n",
    f"{datetime.now().isoformat()} INFO final_val_loss=2.000000\n",
    f"{datetime.now().isoformat()} INFO final_bleu=0.150000\n",
]

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Wrote demo training log: {path}")


