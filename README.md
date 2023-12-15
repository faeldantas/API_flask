# API de Previsão de Emoções com Processamento de Linguagem Natural (PLN) e NLTK

Bem-vindo à API de Previsão de Emoções, uma aplicação desenvolvida em Python utilizando o framework Flask e a biblioteca NLTK para processamento de linguagem natural. Esta API permite prever emoções com base em texto de entrada, utilizando um modelo de machine learning treinado para análise de sentimentos.

## Como Utilizar a API

### Requisitos

Certifique-se de ter o Python instalado em sua máquina. Você pode instalar as dependências necessárias executando o seguinte comando:

```bash
pip install -r requirements.txt
```

### Inicialização
Para iniciar a API, execute o seguinte comando:
```bash
python app.py
```
A API estará disponível em 'http://localhost:5000'.

##Exemplo de Uso
Envie uma solicitação POST para 'http://localhost:5000/predict' com o seguinte corpo:
```json
{
  "text": "Eu estou muito feliz com os resultados do projeto!"
}
```




