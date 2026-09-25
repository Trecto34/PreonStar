# Interference Search for Q36

Implementation of the **Interference Search** paradigm (based on research by [Bad Theory Labs](https://github.com/Badtheorylabs/interference-search)) tailored for **Q36** and local LLMs on the AMD BC-250 / Apple Silicon.

---

## 🔬 Conceito

Diferente do raciocínio linear dos LLMs (*Chain-of-Thought* contínuo), o **Interference Search** raciocina sobre **estados explícitos** com busca concorrente:

1. **Expansão Concorrente**: Vários ramos geram hipóteses de ação/código ao mesmo tempo.
2. **Execução no Ambiente**: Compiladores, interpretadores e suítes de teste avaliam cada ramo de forma determinística.
3. **Interferência Construtiva (Merge)**: Caminhos distintos que produzem o mesmo comportamento lógico/testes colapsam em um único estado, acumulando os pesos dos caminhos pais (`1 / (rank + 1)`).
4. **Interferência Destrutiva (Poda)**: Erros de sintaxe ou caminhos sem saída são cancelados imediatamente.
5. **Avanço Sincronizado**: Os $K$ melhores estados avançam juntos para o próximo passo.

---

## 📁 Estrutura no Worktree

```
q36-interference/
├── interference/
│   ├── core.py                   # Loop universal de busca com merge e cancelamento
│   ├── cli.py                    # Visualizador e benchmark interativo
│   ├── domains/
│   │   ├── base.py               # Contrato abstrato de domínio
│   │   ├── code_domain.py        # Síntese e reparo de código Python com sandbox
│   │   └── countdown_domain.py   # Benchmark exato Countdown do paper
│   └── backends/
│       ├── base.py               # Interface de LLM
│       ├── q36_backend.py        # Conexão nativa com q36 / q36-server
│       └── mock_backend.py       # Backend determinístico para testes e CI
└── tests/
    └── test_interference.py      # Suíte de testes unitários (100% passando)
```

---

## 🚀 Como Executar

### 1. Benchmark Countdown (Ground Truth Matemático)
```bash
python3 -m interference.cli --domain countdown
```

### 2. Raciocínio de Código (Mock / Teste Rápido)
```bash
python3 -m interference.cli --domain code --backend mock
```

### 3. Raciocínio de Código com o Q36 Local (Ternary-Bonsai no Vulkan)
Com o `q36-server` ativo (ou chamando `./q36` diretamente):
```bash
python3 -m interference.cli --domain code --backend q36
```

### 4. Executar Testes Unitários
```bash
python3 -m unittest tests/test_interference.py
```
