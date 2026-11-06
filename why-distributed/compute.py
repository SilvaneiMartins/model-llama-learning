
import torch
from models import GPU_PEAK_FLOPS, MFU, MODELS, REPORTED_GPU_HOURS, SEQ_LENGTH, TOKENS
from torch.utils.flop_counter import FlopCounterMode

from dtrain.flops import (
    attention_flops_per_token,
    embedding_params,
    matmul_params,
    norm_params,
)
from dtrain.memory import num_params
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn

def measured(fn) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()

def check_on_small_model():
    config = LlamaConfig(max_position_embeddings=128)
    seq, batch = 128, 2
    tokens = seq * batch
    model = Llama(config)
    ids = torch.randint(0, config.vocab_size, (batch, seq + 1))
    total = measured(lambda: loss_fn(model(ids[:, :-1]), ids[:, 1:]).backward())
    
    print(f"\n5. Verificação com torch.utils.flop_counter em um Llama de {num_params(config):,} parâmetros")
    print(f"   {'FLOPs de multiplicação matricial por token, medidos':<42} {total // tokens:>10,}")
    print(
        f"   {f'6 × {matmul_params(config):,} pesos de multiplicação matricial':<42} {6 * matmul_params(config):>10,}"
    )

    h, heads = config.hidden_size, config.num_attention_heads
    q, k, v = (torch.randn(batch, heads, seq, h // heads, requires_grad=True) for _ in range(3))

    def attention():
        scores = torch.softmax(q @ k.transpose(-1, -2) / (h // heads) ** 0.5, dim=-1)
        (scores @ v).sum().backward()

    print(
        f"   {'atenção por camada por token, medida':<42} {measured(attention) // tokens:>10,}"
    )
    print(f"   {f'12 × h {h} × seq {seq}':<42} {12 * h * seq:>10,}")


def main():
    config = MODELS["Llama 3 8B"]
    params = num_params(config)
    matmul = matmul_params(config)
    attention = attention_flops_per_token(config, SEQ_LENGTH)
    layers, h = config.num_hidden_layers, config.hidden_size

    print("1. Uma multiplicação de matrizes")
    print("   (m × k) @ (k × n): cada uma das m·n saídas exige k multiplicações e k adições")
    print("   → 2·m·k·n FLOPs. Por token, cada peso da multiplicação matricial custa 2: uma multiplicação e uma adição")

    print("\n2. Passagem direta, por token, Llama 3 8B")
    print(f"   todos os parâmetros                  {params:>16,}")
    print(f"   − embedding (consulta, sem cálculo)   {embedding_params(config):>16,}")
    print(f"   − pesos RMSNorm (elemento a elemento) {norm_params(config):>16,}")
    print(f"   = pesos usados nas multiplicações    {matmul:>16,}")
    print(f"   passagem direta: 2 × {matmul:,} = {2 * matmul / 1e9:.1f} GFLOPs por token")

    print("\n3. Retropropagação = 2 × passagem direta")
    print("   y = x·W retorna por duas multiplicações matriciais do mesmo tamanho:")
    print("   dx = dy·Wᵀ, para continuar a retropropagação, e dW = xᵀ·dy, para atualizar W")
    print(f"   passagem direta + retropropagação: 6 × {matmul:,} = {6 * matmul / 1e9:.1f} GFLOPs por token")

    print("\n4. A atenção adiciona operações que não possuem pesos")
    print("   a consulta de cada token é comparada com todas as chaves da sequência: 2·h·seq por camada")
    print("   e combina os valores com essas pontuações: 2·h·seq por camada")
    print("   4·h·seq na passagem direta, × 3 com retropropagação = 12·h·seq por camada por token")
    print(f"   seq {SEQ_LENGTH}: 12 × {layers} × {h} × {SEQ_LENGTH} = {attention / 1e9:.1f} GFLOPs")
    print(f"   isso representa +{attention / (6 * matmul):.0%} além das multiplicações matriciais")
    print("   (calculado sem máscara causal, conforme a convenção usual; a máscara pode evitar cerca da metade)")

    check_on_small_model()

    per_token = 6 * matmul + attention
    rule = 6 * params * TOKENS
    print(f"\n6. O treinamento completo, {TOKENS / 1e12:.0f}T tokens")
    print(f"   regra prática 6·N·D:  6 × {params / 1e9:.2f}e9 × {TOKENS:.1e} = {rule:.2e} FLOPs")
    step = per_token * TOKENS
    print(f"   passo a passo:         {per_token / 1e9:.1f}e9 × {TOKENS:.1e} = {step:.2e}")
    print("   A regra contabiliza o embedding, que não realiza cálculos, e desconsidera a atenção.")
    print(f"   Com seq {SEQ_LENGTH}, o cálculo passo a passo é {step / rule - 1:.0%} maior.")
    print("   O tempo abaixo utiliza a regra, a convenção padrão, portanto representa um limite inferior.")

    effective = GPU_PEAK_FLOPS * MFU
    seconds = rule / effective
    hours = seconds / 3600
    print("\n7. Conversão de FLOPs em tempo")
    print(f"   desempenho máximo H100 bf16 denso    {GPU_PEAK_FLOPS / 1e12:.0f} TFLOP/s")
    print(f"   MFU {MFU:.0%}: os FLOPs do modelo são executados a {effective / 1e12:.0f} TFLOP/s")
    print(f"   {rule:.2e} / {effective:.2e} = {seconds:.2e} s = {hours / 1e6:.2f}M horas de GPU")
    print(f"   uma GPU: {hours / 24 / 365:.0f} anos")
    for gpus in (1024, 16384):
        print(f"   {gpus:,} GPUs, escalabilidade perfeita: {hours / gpus / 24:.1f} dias")
    reported = REPORTED_GPU_HOURS["Llama 3 8B"]
    print(f"   A Meta informa {reported / 1e6:.2f}M horas de GPU (ficha técnica do Llama 3.1)")
    print(f"   {reported / hours:.1f}× esta estimativa, que desconsidera a atenção e todas as interrupções")


if __name__ == "__main__":
    main()
