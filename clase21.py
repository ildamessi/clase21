#Importar Librerias o Herramientas para facilitar el trabajo
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Cargar los datos
dataset = pd.read_csv("datos_estudiantes_1000.csv")

# Ver los primeros cinco registros del dataset
primerosCincoRegistros = dataset.head()

# Visualizar las correlaciones
sns.pairplot(dataset)
plt.suptitle("Relaciones entre variables", y=1.2)
plt.show()

# Separar variables independientes y dependientes
x = dataset[['Horas_Estudio','Asistencia', 'Tareas']]
y = dataset['Nota_Final']

# Dividir en conjunto de entrenamiento y conjunto de testing
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.20, random_state=0)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(x_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(x_test)

# Evaluar el modelo
print("Error cuadratico medio (MSE):", mean_squared_error(y_test, y_pred))
print("Coeficiente R^2:",r2_score(y_test, y_pred))

# Comparar valores reales Vs predecidos

plt.figure(figsize=(6,8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Nota Real')
plt.ylabel('Nota Predecida')
plt.title("Comparación de Notas Reales Vs. Predecidas")
plt.plot([0,100],[0,100],'r--')
plt.grid(True)
plt.show()

