import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\alide\PycharmProjects\data-science-intro\data/column_2C_weka.csv")
def  show_data(): # выводим краткую информацию о данных
    print('общая информация о данных:', end = '\n'*2)
    print(df.info(), end = '\n'*2)
    print(df.head(n = 5), end = '\n'*2)
    print('уникальные классы: {}'.format(df['class'].unique()), end = '\n'*2)

def vector_classes():
    return df["class"].apply(lambda x: 1 if x == "Abnormal" else 0)

def plot_data(): # визуализируем данные на плоскости
    f, axel = plt.subplots(1,1)
    f.set(facecolor = '0.5', figwidth = 16, figheight = 8)
    axel.set(facecolor = 'black')
    def get_color1(i):
        Color = ["Lime", "Red"]
        return Color[i]
    a = df.iloc[:, :6].to_numpy()
    scaler_a = StandardScaler()
    a = scaler_a.fit_transform(a)
    pca = PCA(n_components = 2)
    x_2d = pca.fit_transform(a)
    y = vector_classes()
    axel.scatter(x_2d[:, 0], x_2d[:, 1], color = [get_color1(i) for i in y], alpha = 0.8)

def outlier_iqr(df, x, threshold): # функция для чистки данных от выбросов, для лучшего обучения модели
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    up_bound = q3 + threshold * iqr
    low_bound = q1 - threshold * iqr
    up = np.where(x < up_bound)
    low = np.where(x > low_bound)
    index = np.intersect1d(up,low)
    return df.iloc[index]

def train_and_estimate_model():
    y = vector_classes()
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :6], y, test_size = 0.2, random_state = 42, stratify = y)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    param_grid = {"n_neighbors": np.arange(1, 51), "p": np.arange(1, 4), "weights": ["uniform", "distance"]}
    knn = KNeighborsClassifier()
    cv = StratifiedKFold(n_splits = 5,shuffle = True)
    grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = cv, scoring = 'f1')
    Q = grid_search.fit(x_train_scaled, y_train)
    print('лучшая метрика f1 на кроссвалидации: {}'.format(Q.best_score_))
    test_predictions = Q.best_estimator_.predict(x_test_scaled)
    print('лучшая метрика f1 на тестовых данных: {}'.format(f1_score(test_predictions, y_test)))
    accuracy = accuracy_score(test_predictions, y_test)
    print('лучшая метрика accuracy на тестовых данных: {}'.format(accuracy))
    accuracy_baseline = y.value_counts(normalize=True).iloc[0]
    print('лучше чем базовый прогноз на: {} %'.format(100*(accuracy - accuracy_baseline)/accuracy_baseline))

if __name__ == '__main__':
    show_data()
    plot_data()
    for i in range(df.columns.size - 1):
        df = outlier_iqr(df, df[df.columns[i]], 1.5)
    train_and_estimate_model()
    plt.show()