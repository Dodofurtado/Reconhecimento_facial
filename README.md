# Reconhecimento Facial com Python

Um sistema simples de reconhecimento facial usando Python, OpenCV e Machine Learning.

## 📋 Requisitos

- Python 3.x
- OpenCV (`opencv-python`)
- scikit-learn
- joblib
- numpy

Para instalar as dependências:
```bash
pip install opencv-python scikit-learn joblib numpy
```

## 🚀 Como usar

### 1. Capturar fotos para treinar

Execute `capturar_foto.py` para criar o dataset:
```bash
python capturar_foto.py
```
- Digite o nome da pessoa
- Pressione 'f' para tirar várias fotos (recomendado: 10+ fotos por pessoa)
- Pressione 'q' para mudar de pessoa
- Pressione 'esc' para sair e treinar o modelo

### 2. Reconhecimento em tempo real

Execute `main.py` para reconhecer rostos:
```bash
python main.py
```
- Enquadre o rosto no oval verde
- Pressione 'q' para tentar reconhecer
- Pressione 'esc' para sair

## 📁 Estrutura do Projeto

- `capturar_foto.py`: Interface para capturar fotos e criar o dataset
- `treinar_rosto.py`: Script para treinar o modelo de ML com as fotos
- `main.py`: Interface de reconhecimento em tempo real
- `dataset_rosto/`: Pasta com as fotos organizadas por pessoa
- `modelo_rosto.joblib`: Modelo treinado

## 🤖 Como funciona

1. **Captura**: O sistema usa a webcam para capturar fotos do rosto.
2. **Dataset**: As fotos são organizadas em pastas com o nome de cada pessoa.
3. **Treinamento**: Um modelo RandomForest é treinado para reconhecer os rostos.
4. **Reconhecimento**: O sistema compara rostos em tempo real com o modelo treinado.

## 📈 Melhorando o Reconhecimento

- Tire várias fotos de cada pessoa
- Capture diferentes ângulos e expressões
- Mantenha boa iluminação
- Centralize bem o rosto no oval verde

## ⚙️ Configurações

- Ajuste o threshold de confiança em `main.py` (padrão: 0.6 ou 60%)
- Modifique o tamanho do oval alterando a variável `tam` nos scripts
- O modelo usa imagens 64x64 em escala de cinza para treinar

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
