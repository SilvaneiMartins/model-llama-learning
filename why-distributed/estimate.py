import math

from models import GPU_MEMORY, GPU_PEAK_FLOPS, MFU, MODELS, REPORTED_GPU_HOURS, TOKENS

from martins_llama.memory import num_params, training_bytes

SECONDS_PER_DAY = 86400

def duration(days: float) -> str:
    return f"{days / 365:,.0f} anos" if days > 365 else f"{days:,.1f} dias"

def main():
    peak = GPU_PEAK_FLOPS / 1e12
    print(f"{TOKENS / 1e12:.0f}T tokens · GPU de 80 GB · {peak:.0f} TFLOP/s bf16 · {MFU:.0%} MFU")
    for name, config in MODELS.items():
        params = num_params(config)
        state = sum(training_bytes(params, fp32_grad_accumulation=True).values())
        flops = 6 * params * TOKENS
        days = flops / (GPU_PEAK_FLOPS * MFU) / SECONDS_PER_DAY
        print(f"\n{name}  {params / 1e9:.1f}B parâmetros")
        print(f"  estado do treinamento       {state / 1e9:>10,.0f} GB")
        print(f"  GPUs apenas para armazenar  {math.ceil(state / GPU_MEMORY):>10,}")
        print(f"  custo computacional         {flops:>10.1e} FLOPs")
        hours = days * 24
        reported = REPORTED_GPU_HOURS[name]
        print(f"  horas de GPU, estimadas     {hours / 1e6:>9.2f}M")
        print(f"  horas de GPU, informadas    {reported / 1e6:>9.2f}M   Ficha técnica do Llama 3.1")
        for gpus in (1, 1024, 16384):
            print(f"  com {gpus:>6,} GPUs          {duration(days / gpus):>15}")


if __name__ == "__main__":
    main()
