# Por que treinamento distribuído?

Quanto de memória é necessário para o treinamento, e quanto tempo uma única GPU levaria? Estes scripts abordam ambas as questões, começando pelo próprio modelo.

O Llama 3 8B é o exemplo principal. Os cálculos também abrangem as configurações de 70B e 405B. Os cálculos para modelos grandes utilizam o _meta device_ do PyTorch; os testes numéricos empregam um modelo pequeno que cabe em um notebook.

Execute os comandos abaixo a partir da raiz do repositório, após [configurar o ambiente](../README.md#run-the-code).

## Acompanhando uma sequência

```bash
uv run python why-distributed/walkthrough.py
uv run python why-distributed/walkthrough.py --text "What happens inside attention?"
```

O script rastreia os IDs de tokens através dos _embeddings_, do mecanismo de atenção (_attention_), da MLP e da projeção de saída. Ele exibe as dimensões (_shapes_) e conta as multiplicações de matrizes, verificando, em seguida, o total por meio do contador de FLOPs do PyTorch.

Na primeira execução, o script baixa o _tokenizer_ do Llama 3 a partir de `NousResearch/Meta-Llama-3-8B`. Ele não realiza o download dos pesos do modelo.

A implementação encontra-se em [`model.py`](../src/martins_llama/model.py): inclui RMSNorm, _rotary position embeddings_, _grouped-query attention_, SwiGLU e a pilha de decodificadores (_decoder stack_). O arquivo [`test_model.py`](../tests/test_model.py) compara _logits_ e gradientes com a biblioteca Transformers, utilizando a mesma configuração reduzida e os mesmos pesos.

## Contagem de parâmetros

```bash
uv run python why-distributed/params.py
uv run python why-distributed/params.py --model "Llama 3 70B"
```

Para a configuração 8B, a contagem é de **8.030.261.248 parâmetros**. Cerca de 70,2% estão na MLP, 16,7% no mecanismo de atenção e 13,1% nos _embeddings_ e na projeção de saída. O RMSNorm acrescenta uma pequena parcela adicional.

O script realiza a contagem com base nas dimensões dos módulos atuais. Os testes verificam essas contagens com base na fórmula presente em [`memory.py`](../src/martins_llama/memory.py).

## Converter parâmetros em memória

```bash
uv run python why-distributed/memory.py
```

A configuração do otimizador Adam com precisão mista utilizada aqui apresenta:

| Estado                                      | Bytes por parâmetro |
| ------------------------------------------- | ------------------: |
| Pesos em bf16                               |                   2 |
| Gradientes em bf16                          |                   2 |
| Pesos principais (_master weights_) em fp32 |                   4 |
| Primeiros e segundos momentos do Adam       |                   8 |
| Buffer de acumulação de gradientes em fp32  |                   4 |
| Total                                       |                  20 |

Para um modelo de 8B (8 bilhões de parâmetros), isso resulta em **160,6 GB de estado de treinamento**, sem contar as ativações e o _overhead_ (sobrecarga) do framework. Esta é uma configuração específica: ao omitir o buffer de acumulação separado em fp32, a estimativa passa a ser de 16 bytes por parâmetro.

O script também demonstra por que a precisão é importante. Subtrair `0,001` de um valor `1,0` representado em bf16 resulta, após o arredondamento, no próprio `1,0`. Manter uma cópia principal (_master copy_) em fp32 permite que essa atualização seja acumulada. Adições sucessivas de pequenos gradientes enfrentam um problema de arredondamento semelhante.

A memória necessária para as ativações depende da arquitetura, do comprimento da sequência, do tamanho do _microbatch_ e da implementação do mecanismo de atenção (_attention_). Os valores de memória para ativações exibidos por este script são estimativas para a configuração especificada.

## Converter computação em tempo

```bash
uv run python why-distributed/compute.py
```

Para uma multiplicação de matrizes `(m × k) @ (k × n)`, a contagem usual é de `2mkn` FLOPs. Aplicado a uma camada linear, isso resulta em aproximadamente dois FLOPs por parâmetro, por token, durante a etapa _forward_ (propagação direta). A etapa _backward_ (propagação reversa) adiciona duas multiplicações de matrizes, elevando o total para aproximadamente seis FLOPs por parâmetro, por token.

O mecanismo de atenção também realiza operações que não estão vinculadas a uma matriz de pesos aprendíveis. O script calcula esse termo separadamente. O arquivo [`test_flops.py`](../tests/test_flops.py) verifica ambos os termos comparando-os com o contador do PyTorch.

A estimativa de tempo de processamento em GPU utiliza:

- `6ND`, considerando 8,03 bilhões de parâmetros e 15 trilhões de tokens. - Um pico de 989 TFLOP/s para a H100 (usando formato bf16).
- 40% de utilização de FLOPs do modelo.

Com base nessas premissas, a estimativa é de cerca de **58 anos em uma única GPU**, ou **20,6 dias em 1.024 GPUs, considerando um escalonamento perfeito**. A aproximação `6ND` omite o termo referente ao mecanismo de atenção. Execuções distribuídas reais também envolvem custos de comunicação e outras sobrecargas.

## Comparação de tamanhos de modelos

```bash
uv run python why-distributed/estimate.py
```

Este comando exibe as estimativas de memória e processamento para modelos de 8B, 70B e 405B, utilizando as mesmas premissas. A quantidade mínima de GPUs considera apenas o estado do treinamento; as ativações e a sobrecarga durante a execução exigem espaço adicional.
