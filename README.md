# Treinamento distribuído

Estou estudando como modelos são treinados utilizando múltiplas GPUs. Este repositório é onde exploro essas questões por meio de código, diagramas e pequenos experimentos.

## 🚀 Como Executar o Projeto

Este projeto utiliza o [uv](https://docs.astral.sh/uv/), um gerenciador de pacotes e ambientes Python extremamente rápido escrito em Rust. Você não precisa criar ou ativar ambientes virtuais manualmente.

### Pré-requisitos

Certifique-se de ter o `uv` instalado em sua máquina:

```bash
# Caso não tenha o uv instalado:
curl -LsSf https://astral.sh | sh
```

### 1. Clonar o Repositório

```bash
git clone https://github.com/SilvaneiMartins/model-llama-learning
cd model-llama-learning
```

### 2. Sincronizar o Ambiente (Criar .venv e Instalar PyTorch)

O comando `uv sync` vai ler as dependências do projeto (incluindo o `torch`), criar a pasta `.venv` automaticamente e instalar tudo na velocidade da luz:

```bash
uv sync
```

_(Nota: O `uv` detectará automaticamente a versão correta do PyTorch configurada no arquivo `pyproject.toml` / `uv.lock`)._

### 3. Executar o Código

Para rodar os scripts garantindo que eles utilizem o ambiente isolado correto, basta prefixar o comando com `uv run`:

```bash
uv run seu_script.py
```

Se o seu script aceitar argumentos de linha de comando para o treinamento distribuído (ex: número de GPUs), você pode passar diretamente:

```bash
uv run torchrun --nproc_per_node=2 seu_script.py
```

---

## 👤 Autor

Desenvolvido por **Silvanei Martins**

- 💼 [LinkedIn](https://www.linkedin.com/in/silvanei-martins-a5412436)
- 🌐 [Site Pessoal](https://silvaneimartins.com.br/)
- 🐱 [GitHub](https://github.com/Store-Sam-Martins)
- 📧 silvaneimartins_rcc@hotmail.com
- 🎥 [YouTube](https://www.youtube.com/@silvaneimartins2487/featured)
- 🐦 [X (Twitter)](https://x.com/SilvaneiMartins)

<a href="https://github.com/SilvaneiMartins">
    <img
        style="border-radius:50%"
        src="https://github.com/SilvaneiMartins.png"
        width="100px;"
        alt="Silvanei Martins"
    />
    <br />
    <sub>
        <b>Silvanei de Almeida Martins</b>
    </sub>
</a>
     <a href="https://github.com/SilvaneiMartins" title="Silvanei martins" >
 </a>
<br />
🚀 Feito com ❤️ por Silvanei Martins
