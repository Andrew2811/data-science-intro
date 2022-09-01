import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv(r"C:\Users\alide\PycharmProjects\data-science-intro\data\flat.csv", sep = '\t')
df.set_index("n", inplace=True)
def  show_data(): # выводим краткую информацию о данных
    print('общая информация о данных:', end = '\n' * 2)
    print(df.info(), end = '\n' * 2)
    print(df.head(n = 5), end = '\n' * 2)

def outlier_iqr(data, x, threshold):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    up_bound = q3 + threshold * iqr
    low_bound = q1 - threshold * iqr
    up = np.where(x < up_bound)
    low = np.where(x > low_bound)
    index = np.intersect1d(up,low)
    return data.iloc[index]

def get_data(df):
    for i in range(1,6):
        df = outlier_iqr(df,df[df.columns[i]],1.5)
    ylog = np.log(df['price'])
    categorial = pd.get_dummies(df['code'], prefix='code')
    df.drop(['price','code'], axis = 1, inplace = True)
    data = pd.concat([df, categorial], axis = 1)
    data.reset_index(drop = True, inplace = True)
    return ylog, data

def metrics_show(ylog_pred, ylog_test): # расчет метрик прогнозной модели
    def mape(y_pred, y_test):  # расчет средней процентной ошибки
        y1 = y_pred
        y2 = y_test
        t = np.mean(np.abs((y1 - y2) / y2))
        return t * 100
    y_pred = np.exp(ylog_pred)
    y_test = np.exp(ylog_test)
    print("средняя процентная ошибка MAPE: {}% ".format(mape(y_pred, y_test)))
    print("средняя квадратичная ошибка MSE: ", metrics.mean_squared_error(y_pred, y_test))
    print("квадратный корень из средней квадратичной ошибки RMSE: ", metrics.mean_squared_error(y_pred, y_test) ** 0.5)
    print("средняя абсолютная ошибка MAE: ", metrics.mean_absolute_error(y_pred, y_test))
    print("R2: ", metrics.r2_score(y_pred, y_test))

def get_train_data(data):
    dfy, dfx = data[0].to_numpy(), data[1].to_numpy() # data = get_data(df)
    scaler = StandardScaler()  # теперь когда данные готовы, строим модель линейной регрессии¶
    x_train, x_test, ylog_train, ylog_test = train_test_split(dfx, dfy, test_size=0.25, random_state=42)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, ylog_train, x_test_scaled, ylog_test

def train_baseline(data):
    ylog_baseline = np.mean(get_train_data(data)[1])  # Базовый прогноз baseline
    ylog_pred_baseline = np.ones_like(get_train_data(data)[3]) * ylog_baseline
    return ylog_pred_baseline

def train(data):
    x_train_scaled, ylog_train, x_test_scaled, ylog_test = get_train_data(data)
    model_regr = LinearRegression()  # Теперь обучим простую линейную модель без регуляризации, и посмотрим на метрики качества прогноза
    model_regr.fit(x_train_scaled, ylog_train)
    ylog_pred_regr = model_regr.predict(x_test_scaled)
    return ylog_pred_regr

def best_train(data):
    x_train_scaled, ylog_train, x_test_scaled, ylog_test = get_train_data(data)
    lasso_cv = LassoCV(cv = 5) # Теперь обучим линейную модель регресии Lasso на кросс-валидации (найдем оптимальный гипперпараметр регуляризации),
    # и посмотрим на метрики качества прогноза, сравним с обычной без регуляризации
    lasso_cv.fit(x_train_scaled, ylog_train)
    ylasso_best_pred = lasso_cv.predict(x_test_scaled) # Делаем прогноз с уже найденным гиперпараметром
    model_lasso_cv = pd.DataFrame({'x': data[1].columns, "w": lasso_cv.coef_})
    model_lasso_cv.set_index('x', drop = True, inplace = True)
    model_lasso_cv.sort_values(by = "w", inplace=True, ascending=False)
    return ylasso_best_pred, model_lasso_cv

def show_estimate_factors(data):
    f, axel = plt.subplots(1, 1) # визуализируем важность признаков: чем больше абсолютное значение у признака - тем он лучше
    # объясняет (предсказывает) линейную регрессию
    f.set(figwidth = 16, figheight = 8)
    axel.bar(best_train(data)[1].index, best_train(data)[1]['w'], color = 'red', label = 'lasso_cv')
    axel.legend()

if __name__ == '__main__':
    show_data()
    data = get_data(df)
    print("Метрики до обучения - оценка базового прогноза:", end='\n' * 2)
    metrics_show(train_baseline(data), get_train_data(data)[3])
    print("Метрики после обучения простой линейной регрессии:", end='\n' * 2)
    metrics_show(train(data), get_train_data(data)[3])
    print("Метрики после обучения lasso_cv модели:", end='\n' * 2)
    metrics_show(best_train(data)[0], get_train_data(data)[3])
    # визуализируем важность признаков
    show_estimate_factors(data)
    plt.show()
