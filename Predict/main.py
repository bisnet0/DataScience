import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
url = 'https://www.gov.br/receitafederal/dados/arrecadacao-estado.csv'
df = pd.read_csv(url, encoding='latin1', sep=';')

# Selecionar as colunas relevantes
df = df[['Ano', 'Mês', 'UF', 'IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO']]

# Verificar valores nulos
print(df.isnull().sum())

# Preencher valores nulos com 0 (se necessário)
df = df.fillna(0)

# Convertendo as colunas numéricas
numeric_columns = ['IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Remover linhas onde houver valores NaN
df = df.dropna()

# Codificar a coluna 'Mês'
month_map = {'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6,
             'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12}
df['Mês'] = df['Mês'].map(month_map)

# Codificar a coluna 'UF'
df = pd.get_dummies(df, columns=['UF'])

# Dividir as features e o label para IMPOSTO SOBRE IMPORTAÇÃO
features_imp = df.drop(columns=['IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO'])
label_imp = df['IMPOSTO SOBRE IMPORTAÇÃO']
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(features_imp, label_imp, test_size=0.2, random_state=42)

# Treinar o modelo KNN para IMPOSTO SOBRE IMPORTAÇÃO
knn_imp = KNeighborsClassifier(n_neighbors=3)
knn_imp.fit(X_train_imp, y_train_imp)

# Fazer previsões e calcular a acurácia para IMPOSTO SOBRE IMPORTAÇÃO
y_pred_imp = knn_imp.predict(X_test_imp)
accuracy_imp = accuracy_score(y_test_imp, y_pred_imp)
print(f'Acurácia do modelo para IMPOSTO SOBRE IMPORTAÇÃO: {accuracy_imp:.2f}')

# Dividir as features e o label para IMPOSTO SOBRE EXPORTAÇÃO
features_exp = df.drop(columns=['IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO'])
label_exp = df['IMPOSTO SOBRE EXPORTAÇÃO']
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(features_exp, label_exp, test_size=0.2, random_state=42)

# Treinar o modelo KNN para IMPOSTO SOBRE EXPORTAÇÃO
knn_exp = KNeighborsClassifier(n_neighbors=3)
knn_exp.fit(X_train_exp, y_train_exp)

# Fazer previsões e calcular a acurácia para IMPOSTO SOBRE EXPORTAÇÃO
y_pred_exp = knn_exp.predict(X_test_exp)
accuracy_exp = accuracy_score(y_test_exp, y_pred_exp)
print(f'Acurácia do modelo para IMPOSTO SOBRE EXPORTAÇÃO: {accuracy_exp:.2f}')

# Análise das tendências de arrecadação
df_grouped = df.groupby('Ano').sum()[['IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO']]

# Plotar as tendências de arrecadação ao longo dos anos
df_grouped.plot(kind='line', figsize=(12, 6))
plt.title('Tendências de Arrecadação ao Longo dos Anos')
plt.ylabel('Arrecadação (R$)')
plt.xlabel('Ano')
plt.grid(True)
plt.show()

# Comparação entre estados (ajustado para variáveis dummy)
# Somar as colunas dummy para obter o total por estado
df_states = df.filter(like='UF_').sum().sort_values(ascending=False)
df_states.plot(kind='bar', figsize=(14, 7))
plt.title('Comparação de Arrecadação por Estado')
plt.ylabel('Arrecadação (R$)')
plt.xlabel('Estado')
plt.grid(True)
plt.show()

# Análise de sazonalidade
df_months = df.groupby('Mês').sum()[['IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO']]
df_months.plot(kind='line', figsize=(12, 6))
plt.title('Sazonalidade nas Receitas de Impostos')
plt.ylabel('Arrecadação (R$)')
plt.xlabel('Mês')
plt.grid(True)
plt.show()

# Análise de correlações
correlation = df[['IMPOSTO SOBRE IMPORTAÇÃO', 'IMPOSTO SOBRE EXPORTAÇÃO']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlação entre as Receitas de Importação e Exportação')
plt.show()
