# Sistema de Reconhecimento Facial com TensorFlow

Projeto de detecção e reconhecimento de faces em tempo real, executado no **Google Colab** com a **webcam do notebook**. O sistema detecta várias faces ao mesmo tempo e identifica cada pessoa pelo nome.

> [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nEqwlFPvwcSdEpEIXgvhbc1EwPI5gQcQ?authuser=1#scrollTo=3-RT70rdRY3c)

---

## Visão geral

O projeto foi desenvolvido para a atividade "Criando um sistema de reconhecimento facial do zero", que pede:

1. Uma **rede de detecção** treinada para detectar faces.
2. Uma **rede de classificação** para classificar a face detectada.
3. Detecção e reconhecimento de **mais de uma face ao mesmo tempo**.

### Pipeline

```
Webcam → Detecção de faces (MTCNN) → Recorte da face (160×160) → Classificação (MobileNetV2) → Nome + confiança na imagem
```

| Etapa | Técnica | Detalhes |
|---|---|---|
| Detecção | **MTCNN** (pacote `mtcnn`, baseado em TensorFlow) | Rede pré-treinada; só aceita detecções com confiança ≥ 0,95 |
| Classificação | **MobileNetV2** com *transfer learning* (TensorFlow/Keras) | Pesos do ImageNet, camada final softmax com uma classe por pessoa |
| Regularização | Aumento de dados e *dropout* | Flip, rotação, zoom, brilho e contraste; dropout de 0,3 |
| Treino | Duas fases | 12 épocas com a base congelada + 8 épocas de *fine-tuning* das últimas 30 camadas |
| Rejeição | Limiar de confiança | Confiança abaixo de 0,80 aparece como "Desconhecido" |

---

## Tecnologias

- Python 3
- TensorFlow / Keras
- MTCNN
- OpenCV
- NumPy e Matplotlib
- Google Colab (acesso à webcam via `google.colab.output.eval_js`)

---

## Como executar

1. Abra o notebook `reconhecimento_facial.ipynb` no Google Colab (pelo botão acima, ou em *File → Upload notebook*).
2. Execute as células **em ordem**.
3. Quando o navegador pedir, **permita o acesso à câmera**.

### Passo a passo do notebook

| Seção | O que faz |
|---|---|
| Configuração | Instala o `mtcnn`, importa as bibliotecas e carrega o detector |
| 1. Funções auxiliares | Captura da webcam, detecção e recorte de faces |
| 2. Coletar fotos | Captura fotos de uma pessoa e salva os recortes em `dataset/<nome>/` |
| 3. Conferir o dataset | Conta as imagens válidas por pessoa e mostra uma amostra |
| 4. Treinar | Treina o classificador e salva o modelo |
| 5. Reconhecer em uma foto | Captura um quadro e reconhece todas as faces nele |
| 6. Reconhecimento ao vivo (opcional) | Processa a webcam quadro a quadro |

### Coletando o dataset

Na seção 2, altere a variável `NOME` e execute a célula **uma vez para cada pessoa**:

```python
NOME = "ana"      # use nomes sem espaços ou acentos
N_FOTOS = 40
```

Recomendações:

- Use **pelo menos 2 pessoas**. Com mais exemplos por pessoa, o resultado melhora.
- Só a pessoa daquele `NOME` deve aparecer na câmera durante a captura.
- Varie o ângulo da cabeça, a expressão e a iluminação.
- Só são salvos os quadros com **exatamente uma face** detectada.
- Opcionalmente, crie uma classe `outros` com rostos variados para reduzir falsos positivos.

---

## Estrutura de arquivos gerados

```
.
├── reconhecimento_facial.ipynb   # notebook principal
├── dataset/                      # gerado na coleta (não versionar)
│   ├── pessoa1/
│   └── pessoa2/
├── face_classifier.keras         # modelo treinado (gerado no treino)
└── class_names.txt               # nomes das classes, na ordem do modelo
```

---

## Parâmetros principais

| Parâmetro | Onde | Padrão | Função |
|---|---|---|---|
| `IMG_SIZE` | Configuração | 160 | Tamanho da face enviada ao classificador |
| `N_FOTOS` | Seção 2 | 40 | Quadros capturados por execução |
| `min_conf` | `detect_faces` | 0,95 | Confiança mínima do MTCNN |
| `LIMIAR` | Seção 5 | 0,80 | Confiança mínima do classificador para dar um nome |
| `n_frames` | `live_recognition` | 100 | Quadros processados no modo ao vivo |

---

## Observações e limitações

- **JavaScript no notebook:** o Python do Colab roda em um servidor remoto e não enxerga a webcam do seu computador. Por isso, a captura usa um trecho curto de JavaScript, executado no navegador, e o resto do pipeline é Python.
- **Desempenho ao vivo:** no Colab, o modo ao vivo roda com poucos quadros por segundo.
- **Poucos dados:** o modelo é treinado com poucas fotos por pessoa. A acurácia de validação pode ficar alta mesmo assim. O teste real é na câmera, com variações de luz e de ângulo.
- **Arquivos temporários:** os arquivos do Colab são apagados quando o ambiente de execução é reiniciado. Baixe `face_classifier.keras` e `class_names.txt` se quiser reutilizar o modelo.
- **Pastas `.ipynb_checkpoints`:** o Colab pode criar essa pasta dentro de `dataset/`. Se aparecer como "arquivo inválido" na seção 3, apague-a com `!find dataset -name ".ipynb_checkpoints" -type d -exec rm -rf {} +`.

---

## Possíveis melhorias

- Usar *embeddings* faciais (por exemplo, FaceNet) para cadastrar novas pessoas sem retreinar a rede.
- Avaliar o modelo em um conjunto de teste separado, com métricas por pessoa e matriz de confusão.
- Testar com um dataset público para medir a acurácia.
- Rodar localmente com `cv2.VideoCapture(0)` para obter mais quadros por segundo.

---

## Referências

- Zhang, K. et al. *Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks* (MTCNN), 2016.
- Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2018.
