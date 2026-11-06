
import math

import torch
from models import GPU_MEMORY, MODELS

from martins_llama.memory import num_params

GB = 1e9

BYTES = [
    ("pesos bf16", 2, "a cópia utilizada pelas multiplicações matriciais na passagem direta e na retropropagação"),
    ("gradientes bf16", 2, "o que a retropropagação produz, um por peso"),
    ("pesos mestres fp32", 4, "bf16 não consegue representar pequenas atualizações, demonstrado abaixo"),
    ("primeiro momento do Adam", 4, "média móvel do gradiente, em fp32"),
    ("segundo momento do Adam", 4, "média móvel do gradiente ao quadrado, em fp32"),
    ("buffer de gradientes fp32", 4, "somar microlotes em bf16 perde valores pequenos, demonstrado abaixo"),
]

def lost_update():
    w = torch.tensor(1.0, dtype=torch.bfloat16)
    step = 1e-3
    print("por que utilizar pesos mestres fp32")
    print(f"  bf16  1.0 - {step} = {(w - step).item()}   a atualização desaparece")
    print(f"  fp32  1.0 - {step} = {(w.float() - step).item():.4f}")
    eps = torch.finfo(torch.bfloat16).eps
    print(f"  bf16 mantém 8 bits de precisão: o próximo valor acima de 1.0 é {1 + eps}")

def stalled_sum():
    bf16 = torch.tensor(0.0, dtype=torch.bfloat16)
    fp32 = torch.tensor(0.0)
    for _ in range(1000):
        bf16 += 1e-3
        fp32 += 1e-3
    print("\npor que utilizar um buffer de gradientes fp32")
    print("  somar 0.001 mil vezes")
    print(f"  bf16  {bf16.item()}   próximo de 0.5, os valores bf16 têm intervalos de 0.0039: + 0.001 é arredondado para o valor anterior")
    print(f"  fp32  {fp32.item():.4f}")

def state():
    total = sum(b for _, b, _ in BYTES)
    print("\nbytes por parâmetro, precisão mista bf16 com Adam")
    for name, size, why in BYTES:
        print(f"  {name:<22} {size:>2}   {why}")
    print(f"  {'total':<22} {total:>2}")
    print("  O artigo ZeRO contabiliza 16: a mesma lista, sem o buffer de gradientes fp32.")
    for name, config in MODELS.items():
        params = num_params(config)
        print(f"\n{name}: {params:,} parâmetros × {total} bytes")
        for part, size, _ in BYTES:
            print(f"  {part:<22} {params * size / GB:>9,.1f} GB")
        print(f"  {'estado do treinamento':<22} {params * total / GB:>9,.1f} GB")
        gpus = params * total / GPU_MEMORY
        print(f"  GPUs de 80 GB para armazenar  {gpus:>9.2f}  → pelo menos {math.ceil(gpus)}")

def activations():
    config = MODELS["Llama 3 8B"]
    layers, h, heads = config.num_hidden_layers, config.hidden_size, config.num_attention_heads
    print("\nativações: tensores mantidos pela passagem direta para a retropropagação")
    print("  por camada, em bf16, para uma sequência: seq · h · (34 + 5 · heads · seq / h) bytes")
    print("  34 · seq · h: entradas das multiplicações matriciais, normalizações, funções de ativação e máscaras de dropout")
    print(
        "  5 · heads · seq²: pontuações de atenção, saída do softmax e máscara de dropout, seq × seq por cabeça"
    )
    print("  (Korthikanti et al. 2022, para uma camada do tipo GPT; o SwiGLU do Llama altera o fator 34)")
    print(f"\n  Llama 3 8B, {layers} camadas, microlote de 1 sequência")
    print(f"  {'seq':>6}  {'pontuações armazenadas':>13}  {'FlashAttention':>15}")
    for seq in (1024, 4096, 8192):
        base = layers * seq * h * 34
        scores = layers * 5 * heads * seq * seq
        print(f"  {seq:>6}  {(base + scores) / GB:>10.1f} GB  {base / GB:>12.1f} GB")
    print("  O FlashAttention nunca armazena a matriz seq × seq: a retropropagação a recalcula em blocos")

def main():
    lost_update()
    stalled_sum()
    state()
    activations()

if __name__ == "__main__":
    main()
