# Strata no Ryzen 9 9950X3D + 2× RTX 3090 — 2026-09-30

O Q2_0 foi o mais rápido na geração de código nesta máquina. O IQ2_XS ajustado ficou em segundo e respondeu melhor em português que o Coder. O Coder teve o melhor resultado no pequeno teste de ferramentas, mas apresentou mistura de idiomas e erro factual em português. Nenhum resultado aqui estabelece confiabilidade ampla para agentes.

## Resultados medidos

Medianas de três respostas por tarefa, após aquecimento, contexto configurado em 32.768 tokens. Código/prosa usam amostragem; a recuperação de informação usa greedy. VRAM inclui o desktop na GPU0. São modelos/quantizações distintos; respostas e aceitação MTP podem diferir.

| Modelo / colocação | Código tok/s | Português tok/s | Capitais tok/s | Primeiro token, código | Pico VRAM GPU0 / GPU1 | Aceitação MTP, código |
|---|---:|---:|---:|---:|---:|---:|
| Q2_0, duas GPUs, mmap | **144,6** | 78,2 | **105,4** | **0,282 s** | 21,20 / 21,55 GiB | 89,5% |
| IQ2_XS, duas GPUs, mmap, min-p 0,7 | 135,5 | **83,9** | 98,2 | 0,326 s | 21,15 / 22,04 GiB | 93,1% |
| Coder IQ1_M, duas GPUs, arena RAM | 108,7 | 70,4 | 85,8 | 0,401 s | 16,72 / 18,47 GiB | 84,8% |
| Coder IQ1_M, GPU1, arena RAM | 95,8 | 61,7 | 80,0 | 0,421 s | 1,06 / 22,49 GiB | 85,3% |

Faixas observadas em código: Q2_0 143,0–146,5; IQ2_XS 133,4–137,9; Coder dual 107,9–111,6; Coder single 95,6–99,1 tok/s. Q2_0 foi 33,0% mais rápido que Coder dual e 6,7% que IQ2_XS ajustado nesta tarefa. Isso não equivale a ganho de qualidade.

O Coder tem metade dos especialistas, mas usa quantização mista mais pesada. Com duas GPUs, os três modelos mantiveram **todos** os especialistas em cache. Assim, menor modelo não implicou kernels mais rápidos. Logs registram 100% de acerto do cache de especialistas em decode.

## Contexto: prefill sem reaproveitamento de KV

Prompts com um código secreto no meio, prefixo diferente por caso; `cache_n=0` comprovado nos resultados. Os tamanhos reais foram 4.010, 16.299 e 28.577 tokens. Nove casos por configuração, todos acertados. A resposta curta de nove tokens serve para verificar recuperação, não para estimar throughput de decode.

| Colocação | TTFT ~4k | TTFT ~16k | TTFT ~28,6k | Prefill ~28,6k |
|---|---:|---:|---:|---:|
| Q2_0 dual | 2,57 s | 6,68 s | 10,02 s | 2.872 tok/s |
| IQ2_XS ajustado dual | 2,65 s | 7,43 s | 10,36 s | 2.785 tok/s |
| Coder dual | 2,45 s | 6,43 s | **9,50 s** | **3.034 tok/s** |
| Coder single | **2,02 s** | 7,02 s | 12,31 s | 2.336 tok/s |

Uma GPU ganhou em TTFT curto do Coder; duas ganharam em decode e contexto longo. Não houve crash/OOM nesses testes. 64k/128k, concorrência, visão e raciocínio longo não foram medidos.

## Qualidade e ferramentas

Dez problemas iniciais do conjunto HumanEval curado do Model Loader, executados com bubblewrap, rede desativada e limite de cinco segundos. Amostragem de código oficial: temperatura 1,0, top-p 0,95, top-k 20, sem presence penalty, seed 42, até 1.024 tokens. **É uma amostra pequena, não uma estimativa robusta de Pass@1.**

| Modelo dual | Código executável | Integridade da ferramenta | Após resultado de sucesso | Seleção automática |
|---|---:|---|---|---|
| Coder | 10/10 | 4/4, inclui streaming | encerrou corretamente | 10/10 chamadas corretas; 10/10 controles sem chamada |
| IQ2_XS ajustado | 10/10 | 4/4, inclui streaming | repetiu a chamada | 10/10 chamadas corretas; 1/10 controles chamou ferramenta |
| Q2_0 | 9/10 | 4/4, inclui streaming | encerrou corretamente | 9/10 chamadas corretas; 10/10 controles sem chamada |

A falha Q2_0 em HumanEval/106 foi lógica: acumulou o fatorial apenas em índices pares. A falha automática de ferramenta foi **comando também escrito em `message.content` junto a `tool_calls`**, não JSON inválido. Ferramentas de teste não foram executadas; somente suas estruturas e o fluxo foram avaliados.

O avaliador inicial omitira funções auxiliares fornecidas pelo próprio enunciado HumanEval. Isso criou uma falha artificial no IQ2_XS em HumanEval/10. O avaliador foi corrigido para preservar imports/helpers do enunciado, e as mesmas respostas foram reexecutadas: Coder e IQ2_XS 10/10. Originais e reavaliações foram preservados.

A repetição de ferramenta do IQ2_XS foi reproduzida com `spec_min_p=0.5` e `0.7` (seeds 42, 43 e 44: 3/3 falhas em cada braço). A seleção automática também repetiu os mesmos 10/10 positivos e 1/10 controles espúrios. O ajuste de MTP não foi a causa isolada observada. Payloads e respostas estão em `results/iq2-tool-floor-check-raw.json`.

Na tarefa em português, Coder misturou inglês/espanhol e descreveu o compressor como bomba de ar. IQ2_XS e Q2_0 produziram texto coerente em português na amostra. Os testes de velocidade limitam a saída a 256 tokens e podem interromper um texto correto antes do fim.

## Ajustes adotados

- Binário do fork `97cb786`, compilado para sm_86; correções #253/#266 já implementadas.
- Duas RTX 3090 24 GiB, **270 W por placa**, driver 610.57.04, CUDA 13.4; sem NVLink, PCIe PHB. Nenhum limite de potência ou driver foi alterado.
- Camadas 24/24; KV `int8`; MTP Q2_0, `--spec 4`; vocabulário CJK; projeção experimental desativada.
- `--expert-cache auto`, `--prefill auto` escolheu chunk 8192, `--vram-reserve-mib 1536`, 15 workers e um núcleo coordenador.
- Q2_0 usa pack canônico com relayout para AVX-512. Q2_0 e IQ2_XS usam `--mmap-experts` para caber com as aplicações abertas. Coder usa arena RAM com registro CUDA completo comprovado no log.
- Q2_0: fração PCIe automática 0,20, `--spec-min-p 0.5`. Coder: PCIe medido 13,4 GB/s → fração automática ~0,28, min-p 0,5. Ganhos das alternativas foram só 1,7% e 3,6%; não adotados.
- IQ2_XS: manter PCIe automático, aumentar **somente** `--spec-min-p` para 0,7. Confirmação alternada de uma variável: medianas greedy 112,3 → 121,0 tok/s (+7,75%). Na comparação amostrada de código: 131,5 → 135,5 tok/s (+3,0%). Não confundir os dois protocolos.
- Não houve motivo para explorar mais workers na colocação dual: todos os especialistas já estavam nas GPUs; o pool CPU não era o gargalo observado.
- Template oficial Qwen fixado por revisão, comparado com o GGUF. O GGUF inclui extensões Unsloth; a cópia isolada usa o oficial. Tokenizadores das variantes têm hashes iguais.
- Sampling padrão non-thinking oficial: temperatura 0,7 / top-p 0,8 / top-k 20 / min-p 0 / presence 1,5 / repetition 1. Calibração: greedy, presence 0; qualidade de código: preset oficial de código acima.

A preparação de ~135 GB de arquivos únicos e dos packs aumentou o uso de swap das aplicações abertas. Nenhuma aplicação do usuário foi encerrada. Antes de medir, downloads e conversões foram concluídos/pausados; as requisições de teste foram seriais. O modo mmap deixa cerca de 39–40 GiB de RAM disponível no cenário medido. A arena do Coder deixou cerca de 12–17 GiB. Não foi feito A/B de arena residente contra mmap nos modelos completos; portanto, não se afirma máximo absoluto sob outra disponibilidade de RAM.

## Reproduzir e inspecionar

Catálogo/perfis/configs isolados: `/home/diogo/models/strata/bench-20260930/`. API: `127.0.0.1:14321`. Nenhum perfil habitual foi sobrescrito. Use o wrapper do benchmark; `XDG_CONFIG_HOME` sozinho não isola o supervisor systemd desta instalação.

```bash
cd /home/diogo/dev/Strata
build-bench/ml instance start qwen3.8-flash-next-q2-0-mtp-strata-dual-mmap-w15-32k
python build-bench/bench.py qwen3.8-flash-next-q2-0-mtp-strata-dual-mmap-w15-32k --mode short --label nova-medicao
build-bench/ml instance stop qwen3.8-flash-next-q2-0-mtp-strata-dual-mmap-w15-32k
```

Os scripts locais estão em `build-bench/`: `make-profile.py`, `bench.py`, `quality.py`, `validate-model.py`, `watch.py`, `report.py`. Os rótulos de medição precisam ser novos para preservar os resultados anteriores. Configuração dos flags nativos está em `configs/<profile>.json`; sampling/thinking compartilhado em `<profile>.shared-settings.json`.

Artefatos: `results/*/raw.jsonl` contém payloads, respostas SSE, tempos e telemetria por placa a cada 0,5 s; `results/*/summary.json`, `results/*tools.json`, logs completos e `hardware-and-build.json`. [`comparison.json`](comparison.json) é a síntese numérica.

O benchmark integrado `model-loader benchmark run --mode llama-bench` passou nos quatro presets do Q2_0. Reutilizou o prompt cache e reportou 64–102 ms de TTFT; esses valores **não** são comparáveis ao prefill integral da tabela. O relatório integrado registra `completionTokens=0` nos agregados apesar de haver geração nos logs; por isso os números principais vêm dos timings/usage brutos verificados.

Fontes: [Strata fork](https://github.com/quantmind-br/Strata), [quants GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF), [modelo oficial](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Revisões, SHA256 dos downloads e template estão nos artefatos locais de preparação/pesquisa.

Ao terminar, todas as instâncias de teste e o proxy isolado foram encerrados. GPUs retornaram a 1.086 MiB (desktop) e 18 MiB, ambas com limite de 270 W.
