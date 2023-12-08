#!/usr/bin/env python
# coding: utf-8

# In[11]:
# Preparar o modelo/ treinar
# Salvar estado com picke
# Implementar funções
# Testar Entrada
# Testar Saída 


import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve


# In[ ]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


# In[ ]:


sentimentos = pd.read_csv('data.csv')
sentimentos.head()


# In[ ]:


sentimentos.info()


# In[ ]:


#print(f"A base de dados tem {5842} sentenças presentes.")
sentimentos['Sentiment'].value_counts()


# In[ ]:


import matplotlib.pyplot as plt

cores_sentimentos = {'positivo': '#3CD36C', 'negativo': '#D33C3C', 'neutro': '#ACACAC'}

plt.figure(figsize=(8, 6))
grafico = sns.countplot(x='Sentiment', data=sentimentos, palette=cores_sentimentos.values())



plt.title('Quantidade de menções em cada sentimentos', fontsize=12)
plt.xlabel('Sentimento', fontsize=10)
plt.ylabel('Quantidade', fontsize=10)

grafico.set_xticklabels(['Positivo', 'Negativo', 'Neutro'])

legenda = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cores_sentimentos[sentimento], markersize=10, label=sentimento.capitalize()) for sentimento in cores_sentimentos.keys()]
plt.legend(handles=legenda, title='Sentimentos', title_fontsize='10')

for barra in grafico.patches:
    altura = barra.get_height()
    grafico.annotate(f'{altura}', (barra.get_x() + barra.get_width() / 2., altura), ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=10)

plt.show()


# In[ ]:


#Aqui é criada uma função para remover as stopwords da língua inglesa,feita a tokenização das sentenças e deixando as palabras em minusculo. Dessa forma eu consigo ter mais informações a respeito da base

#com essas modificações eu crio uma coluna na base de dados e ponho elas no meu dataframe para deixar guardado caso eu precise

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return filtered_words

# Aplicar pré-processamento aos dados
sentimentos['texto_processado'] = sentimentos['Sentence'].apply(preprocess_text)
sentimentos['qtd_palavras'] = sentimentos['texto_processado'].apply(len)


# In[ ]:


sentimentos.head()


# In[ ]:


#Saber a quantidade média de palavras por sentença
media_palavras = sentimentos['qtd_palavras'].mean()
print(f'A quantidade média de palavras por setenças é de {media_palavras:.2f}, o que representa em média 11 palavras por sentenças.')


# In[ ]:


#Qual a setença que possui a maior quantidade de palavras?
qtd_palavras = list(sentimentos['qtd_palavras'])
maior_frase = max(qtd_palavras)
menor_frase = min(qtd_palavras)

qtd_palavras.index(maior_frase)
qtd_palavras.index(menor_frase)


print(sentimentos[['Sentence','Sentiment']].iloc[qtd_palavras.index(maior_frase)])
print(sentimentos[['Sentence','Sentiment']].iloc[qtd_palavras.index(menor_frase)])


# In[ ]:


#Observando a distribuição da quantidade de palavras por sentença de forma agregada

word_count_column = 'qtd_palavras'


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

# Plotagem do histograma
ax1.hist(sentimentos[word_count_column], bins=20, color='skyblue', edgecolor='black')
ax1.set_xlabel('Quantidade de Palavras por Sentença')
ax1.set_ylabel('Número de Sentenças')
ax1.set_title('Distribuição da Quantidade de Palavras por Sentença')


mean_value = sentimentos[word_count_column].mean()
median_value = sentimentos[word_count_column].median()
ax1.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Média ({mean_value:.2f})')
ax1.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Mediana ({median_value:.2f})')
ax1.legend()

# Plotagem do boxplot
boxplot = ax2.boxplot(sentimentos[word_count_column], vert=True, patch_artist=True)
for patch in boxplot['boxes']:
    patch.set_facecolor('skyblue')
ax2.set_xticklabels('')
ax2.set_ylabel('Quantidade de Palavras por Sentença')
ax2.set_title('Boxplot da Quantidade de Palavras por Sentença')

plt.show()


# In[ ]:


#Gráfico de barras para saber as 15 palavras que mais apareceram

def plot_most_common_words(df, column_name, num_words=10):
    # Tokenização e contagem de palavras
    all_words = [word.lower() for words in df[column_name] for word in words]
    freq_dist = FreqDist(all_words)
    
    # Seleção das palavras mais comuns
    most_common_words = freq_dist.most_common(num_words)
    
    # Plotagem do gráfico
    plt.figure(figsize=(18, 6))
    bars = plt.bar(*zip(*most_common_words))
    
    # Adicionar rótulos às barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval), ha='center', va='bottom')
    
    plt.xlabel('Palavras')
    plt.ylabel('Frequência')
    plt.title(f'Top {num_words} Palavras Mais Repetidas')
    plt.show()

# Exemplo de uso
plot_most_common_words(sentimentos, 'texto_processado', num_words=15)


# In[ ]:


sia = SentimentIntensityAnalyzer()
sentimentos['sentiment_score'] = sentimentos['Sentence'].apply(lambda x: sia.polarity_scores(x)['compound'])
print("Média de pontuação de sentimento:", sentimentos['sentiment_score'].mean())


# In[ ]:


sentimentos.head()


# In[ ]:


from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')


#É aplicada a lematização para verificar se quando aplicarmos o modelo fica melhor, mas depois não teve muitas mudanças

def preprocess_dataframe(df, text_column, new_column_name='preprocessed_text'):
 
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))


    df[new_column_name] = df[text_column].apply(lambda sentence: ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]))

    return df


# In[ ]:


sentimentos2 = preprocess_dataframe(sentimentos,'Sentence','texto_processado2')
sentimentos2.head()


# In[ ]:


#Agora usando 8 algoritmos para ver qual o melhor que pode contribuir 
X_train, X_test, y_train, y_test = train_test_split(sentimentos2['texto_processado2'], sentimentos2['Sentiment'], test_size=0.3,random_state=42)

# Vetorização usando TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)

# Criação dos modelos
models = {
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Ridge Classifier': RidgeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Avaliação de cada modelo usando validação cruzada
model_accuracies = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_features, y_train, cv=5, scoring=make_scorer(accuracy_score))
    model_accuracies[model_name] = cv_scores.mean()
    print(f'{model_name}: Acurácia média na validação cruzada: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')

# Escolhe o modelo com a maior acurácia média na validação cruzada
best_model = max(model_accuracies, key=model_accuracies.get)
best_model_instance = models[best_model]

# Exibe as acurácias de cada modelo
print("\nAcurácias de cada modelo:")
for model_name, accuracy in model_accuracies.items():
    print(f'{model_name}: {accuracy:.2f}')


# In[ ]:


best_model_instance.fit(X_train_features, y_train)
X_test_features = vectorizer.transform(X_test)
predictions = best_model_instance.predict(X_test_features)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


print(f'\nMelhor modelo: {best_model}')
print(f'Acurácia nos dados de teste: {accuracy:.2f}')
print('Matriz de Confusão:')
print(conf_matrix)
print('Relatório de Classificação:')
print(classification_rep)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(sentimentos2['texto_processado2'], sentimentos2['Sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)

logistic_regression_model = LogisticRegression(class_weight='balanced',solver='newton-cg',multi_class = 'multinomial')

logistic_regression_model.fit(X_train_features, y_train)


X_test_features = vectorizer.transform(X_test)

predictions = logistic_regression_model.predict(X_test_features)



feature_names = vectorizer.get_feature_names_out()
coeficients_per_class = logistic_regression_model.coef_

coef_dfs = []
for i, class_coeficients in enumerate(coeficients_per_class):
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': class_coeficients})
    coef_df['Class'] = f'Class_{i}'
    coef_dfs.append(coef_df)

final_coef_df = pd.concat(coef_dfs)


sorted_coef_df = final_coef_df.sort_values(by='Coefficient', ascending=False)
sorted_coef_df.head(10)


# In[ ]:


def plot_learning_curve_with_metrics(estimator, X, y, cv=None, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
  
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(14, 8))


    plt.subplot(2, 2, 1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="orange", label="Cross-validation Score")

    plt.title("Curvas de aprendizdo")
    plt.xlabel("Amostras de Treino")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)

    
    plt.subplot(2, 2, 2)

    
    estimator.fit(X, y)

    
    y_pred = estimator.predict(X)

   
    print("Classification Report:")
    print(classification_report(y, y_pred))

    conf_matrix = confusion_matrix(y, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title("Matriz de confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")

    accuracy = accuracy_score(y, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

    plt.tight_layout()
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(sentimentos2['texto_processado2'], sentimentos2['Sentiment'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_features = vectorizer.fit_transform(X_train)

plot_learning_curve_with_metrics(logistic_regression_model, X_train_features, y_train, cv=5, scoring='accuracy')


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ef73d612-bd8d-4eca-82d3-86bec2548c9e' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
