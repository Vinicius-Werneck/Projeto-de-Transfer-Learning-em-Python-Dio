# Classificação de Gatos e Cachorros com TensorFlow

Este projeto utiliza a biblioteca TensorFlow para construir um modelo de aprendizado profundo capaz de classificar imagens de gatos e cachorros. O dataset utilizado é o **Cats and Dogs Dataset**, fornecido pela Microsoft.

---

## **Descrição do Projeto**

O projeto é composto pelas seguintes etapas:

1. **Configuração do Dataset**:
   - Download automático do dataset.
   - Extração e organização dos dados em pastas separadas para gatos e cachorros.

2. **Limpeza e Correção de Imagens**:
   - Verificação e remoção de imagens corrompidas.
   - Redimensionamento das imagens para um tamanho uniforme de 160x160 pixels.

3. **Pré-processamento dos Dados**:
   - Divisão do dataset em conjuntos de treinamento e validação (80/20).
   - Normalização dos pixels para valores entre 0 e 1.

4. **Construção do Modelo**:
   - Utilização do modelo pré-treinado **MobileNetV2** como base (congelado para transfer learning).
   - Adição de camadas personalizadas para classificação binária (gatos e cachorros).

5. **Treinamento e Avaliação**:
   - Treinamento por 1 época (configurável).
   - Avaliação do desempenho no conjunto de validação.
   - Plotagem da acurácia durante o treinamento.

---

## **Pré-requisitos**

- Python 3.7 ou superior
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

---

## **Como Executar o Projeto**

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt

2. **Execute o script principal:**:
   ```bash
   python main.py

---

## **Arquitetura do Modelo**

- Modelo base: MobileNetV2 (pré-treinado no ImageNet)
- Camadas adicionais:
  1. GlobalAveragePooling2D: Reduz as dimensões das features extraídas.
    2. Dense: Camada final com ativação softmax para classificação em 2 classes.

---

## **Resultados**

Após o treinamento por 10 épocas:

- Acurácia no conjunto de validação: 80-90% (dependendo do dataset e dos parâmetros).

---


   

