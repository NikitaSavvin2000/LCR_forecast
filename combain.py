import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go

from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers



tf.keras.backend.clear_session()

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

'2023-06-14'
'2023-06-20'
'2023-06-21'
'2023-06-22'
'2023-11-11'
'2023-11-18'
'2023-11-19'
'2023-11-25'
'2023-11-26'
date_for_test = '2023-12-17'
params_file = '/Users/nikitasavvin/Desktop/Учеба/LCR_forecast/params.yaml'


def replace_zeros_with_average(df, column):
    values = df[column].values  # Получаем значения колонки как numpy массив
    for i in range(len(values)):
        if values[i] == 0:
            # Получаем предыдущее значение, если оно существует
            prev_value = values[i - 1] if i > 0 else None
            # Получаем следующее значение, если оно существует
            next_value = values[i + 1] if i < len(values) - 1 else None

            # Рассчитываем среднее из существующих значений
            neighbors = [v for v in [prev_value, next_value] if v is not None]
            values[i] = np.mean(neighbors) if neighbors else 0  # Если соседей нет, оставляем 0

    df[column] = values  # Обновляем колонку в DataFrame
    return df

class TimeNormalization:

    def __init__(self, col_time, col_target):
        self.min_year = 1900
        self.max_year = 2027
        self.min_weak = 1
        self.max_weak = 52
        self.min_day_of_weak = 0
        self.max_day_of_weak = 6
        self.min_minute = 0
        self.max_minute = 59
        self.min_second = 0
        self.max_second = 59
        self.min_hour = 0
        self.max_hour = 23
        self.scaler = MinMaxScaler()
        self.col_time = col_time
        self.col_target = col_target

    def normalize_column(self, value_series, min_val, max_val):
        """
        Нормировка колонки с использованием Min-Max Scaling.

        Parameters:
        - value_series: pd.Series, колонка значений для нормировки.
        - min_val: float, минимальное значение для нормировки.
        - max_val: float, максимальное значение для нормировки.

        Returns:
        - pd.Series, нормированные значения.
        """
        normalized = value_series.apply(
            lambda x: "None" if pd.isna(x) else (x - min_val) / (max_val - min_val)
        )

        return normalized

    def inverse_normalize_column(self, normalized_value, min_val, max_val):
        """
        Обратная нормировка колонки.

        Parameters:
        - normalized_series: pd.Series, колонка нормированных значений.
        - min_val: float, минимальное значение для обратной нормировки.
        - max_val: float, максимальное значение для обратной нормировки.

        Returns:
        - pd.Series, оригинальные значения.
        """
        if pd.isna(normalized_value):
            original = "None"
        else:
            original = normalized_value * (max_val - min_val) + min_val

        return original

    def check_difrent_years(self):
        if self.min_year != self.max_year:
            return True
        else:
            return False

    def meta_date(self, df):

        df_with_meta = df.copy()
        df_with_meta[self.col_time] = pd.to_datetime(df_with_meta[self.col_time])

        df_with_meta.set_index(self.col_time, inplace=True)
        df_with_meta['year'] = df_with_meta.index.year

        df_with_meta['week'] = df_with_meta.index.isocalendar().week
        df_with_meta['day_of_week'] = df_with_meta.index.dayofweek
        df_with_meta['hour'] = df_with_meta.index.hour

        df_with_meta['minute'] = df_with_meta.index.minute
        df_with_meta['second'] = df_with_meta.index.second
        df_with_meta['hour_sin'] = np.sin(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['hour_cos'] = np.cos(2 * np.pi * df_with_meta['hour'] / 24)
        df_with_meta['day_of_week_sin'] = np.sin(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['day_of_week_cos'] = np.cos(2 * np.pi * df_with_meta['day_of_week'] / 7)
        df_with_meta['week_sin'] = np.sin(2 * np.pi * df_with_meta['week'] / 52)
        df_with_meta['week_cos'] = np.cos(2 * np.pi * df_with_meta['week'] / 52)

        return df_with_meta

    def df_normalize_with_meta(self, df):

        df[self.col_target] = df[self.col_target].astype(float)
        min_val = df[self.col_target].min() * 1.2
        max_val = df[self.col_target].max() * 1.2

        df_with_meta = self.meta_date(df)
        normalized_dates = []
        for index, date in df_with_meta.iterrows():
            if self.check_difrent_years:
                year_norm = (date['year'] - self.min_year) / (self.max_year - self.min_year)
            else:
                year_norm = 1
            weak_norm = ((date['week']) - self.min_weak) / (self.max_weak - self.min_weak)
            day_of_weak_norm = (
                                       date['day_of_week'] - self.min_day_of_weak) / (
                                       self.max_day_of_weak - self.min_day_of_weak
                               )
            minute_norm = (date['minute'] - self.min_minute) / (self.max_minute - self.min_minute)
            second_norm = (date['second'] - self.min_second) / (self.max_second - self.min_second)
            hour_norm = (date['hour'] - self.min_hour) / (self.max_hour - self.min_hour)
            normalized_date = [date[self.col_target], year_norm, weak_norm, day_of_weak_norm, hour_norm, minute_norm,
                               second_norm,
                               date['hour_sin'], date['hour_cos'], date['day_of_week_sin'], date['day_of_week_cos'],
                               date['week_sin'], date['week_cos']]
            normalized_dates.append(normalized_date)

        normalized_df = pd.DataFrame(normalized_dates,
                                     columns=[self.col_target, 'year', 'week', 'day_of_week', 'hour', 'minute',
                                              'second',
                                              'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                                              'week_sin', 'week_cos', ]
                                     )
        normalized_df[self.col_target] = self.normalize_column(normalized_df[self.col_target], min_val, max_val)

        normalized_df = normalized_df.dropna()
        return normalized_df, min_val, max_val

    def df_denormalize_with_meta(self, df, min_val, max_val):
        print('Work fun')
        df = df.sort_values(by=['year', 'week', 'day_of_week', 'hour', 'minute'], ascending=True)

        def _convert_date(date_str):
            parts = date_str.split()
            year, week, day_of_week = map(int, parts[0].split('-'))
            hour, minute, second = map(int, parts[1].split(':'))
            start_of_week = datetime.strptime(f'{year}-{week}-1', '%Y-%W-%w')
            target_date = start_of_week + timedelta(days=day_of_week)
            target_date = target_date.replace(hour=hour, minute=minute, second=second)
            return target_date.strftime('%Y-%m-%d %H:%M:%S')

        denormalized_dates = []
        for index, date in df.iterrows():
            year_denorm = date['year'] * (self.max_year - self.min_year) + self.min_year
            weak_denorm = date['week'] * (self.max_weak - self.min_weak) + self.min_weak
            day_of_weak_denorm = date['day_of_week'] * (
                    self.max_day_of_weak - self.min_day_of_weak) + self.min_day_of_weak
            minute_denorm = date['minute'] * (self.max_minute - self.min_minute) + self.min_minute
            second_denorm = date['second'] * (self.max_second - self.min_second) + self.min_second
            hour_denorm = date['hour'] * (self.max_hour - self.min_hour) + self.min_hour
            denormalized_date = [date[self.col_target], year_denorm, weak_denorm, day_of_weak_denorm, hour_denorm,
                                 minute_denorm, second_denorm,
                                 date['hour_sin'], date['hour_cos'], date['day_of_week_sin'], date['day_of_week_cos'],
                                 date['week_sin'], date['week_cos'],
                                 ]
            denormalized_dates.append(denormalized_date)

        for i in range(len(denormalized_dates)):
            denormalized_dates[i][0] = self.inverse_normalize_column(denormalized_dates[i][0], min_val, max_val)

        denormalized_df = pd.DataFrame(denormalized_dates,
                                       columns=[self.col_target, 'year', 'week', 'day_of_week', 'hour', 'minute',
                                                'second',
                                                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                                                'week_sin', 'week_cos', ]
                                       )

        #TODO: Это костыль нужно убрать!!! (проблема что в колонке hout приходит дробное значение к примеру 12.99 и без строк ниде оно округляется до 12
        import math
        denormalized_df['hour'] = denormalized_df['hour'].apply(lambda x: math.ceil(x))
        denormalized_df['week'] = denormalized_df['week'].apply(lambda x: math.ceil(x))
        denormalized_df['day_of_week'] = denormalized_df['day_of_week'].apply(lambda x: math.ceil(x))
        denormalized_df['minute'] = denormalized_df['minute'].apply(lambda x: math.ceil(x))
        denormalized_df['year'] = denormalized_df['year'].apply(lambda x: math.ceil(x))

        denormalized_df['time_str'] = (denormalized_df['year'].astype(str) + '-' +
                                       denormalized_df['week'].astype(str) + '-' +
                                       denormalized_df['day_of_week'].astype(str) + ' ' +
                                       denormalized_df['hour'].astype(str) + ':' +
                                       denormalized_df['minute'].astype(str) + ':' +
                                       denormalized_df['second'].astype(int).astype(str))

        denormalized_df[self.col_time] = denormalized_df['time_str'].apply(_convert_date)
        return denormalized_df



class SaveBestWeights(Callback):
    def __init__(self):
        super(SaveBestWeights, self).__init__()
        self.best_weights = None
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is None:
            return
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()


def calculate_metrics(y_true, y_pred):
    y_true_mean = y_true.mean()
    # Расчет RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Расчет R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Расчет MAE
    mae = mean_absolute_error(y_true, y_pred)

    # Расчет MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Расчет WMAPE
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    return rmse, r2, mae, mape, wmape

def split_sequence(sequence, n_steps, horizon):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        out_end_ix = end_ix + horizon
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



def create_x_input(df_train, n_steps):
    df_input = df_train.iloc[len(df_train) - n_steps:]
    x_input = df_input.values
    return x_input


def make_predictions(x_input):
    x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, lag, n_features)), dtype=tf.float32)
    predict_values = model.predict(x_input_tensor, verbose=0)
    return predict_values[0]  # Возвращаем предсказания для горизонта



def calc_lcr(previous_val, cur_val):
    if previous_val == 0:
        return cur_val  # Относительно 0 изменение равно текущему значению в процентах

    percentage_change = ((cur_val - previous_val) / abs(previous_val))
    return percentage_change


def make_predictions_lcr(x_input, x_future):
    predict_values = []
    x_future_len = len(x_future)

    for i in range(x_future_len):
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, lag, n_features)), dtype=tf.float32)
        privios_val = x_input_tensor[0][-1][0]
        y_predict = model.predict(x_input_tensor, verbose=0)
        cur_val = y_predict[0][0]
        lcr = calc_lcr(previous_val=privios_val, cur_val=cur_val)
        predict_values.append(y_predict)
        x_input = np.delete(x_input, (0), axis=1)
        future_lag = x_future[0]
        x_future = np.delete(x_future, 0, axis=0)
        future_lag[0] = y_predict
        x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)
        x_input[0][-1][-1] = lcr

    return predict_values


def make_predictions_lcr_2(x_input, x_future, points_per_call):
    """
    Выполняет рекурсивный прогноз на основе x_input и x_future.

    :param x_input: Входные данные для модели (numpy array с размерностью (1, lag, n_features)).
    :param x_future: Дополнительные данные, используемые для обновления lag (numpy array с размерностью (x_future_len, n_features)).
    :param points_per_call: Количество точек, которое модель возвращает за один вызов.
    :return: Список прогнозов длиной x_future_len.
    """
    predict_values = []
    x_future_len = len(x_future)  # Общее количество точек для предсказания
    remaining_horizon = x_future_len

    while remaining_horizon > 0:
        # Количество точек для текущего вызова модели
        current_points_to_predict = min(remaining_horizon, points_per_call)

        # Преобразуем x_input в тензор
        x_input_tensor = tf.convert_to_tensor(x_input.reshape((1, x_input.shape[1], x_input.shape[2])), dtype=tf.float32)

        y_predict = model.predict(x_input_tensor, verbose=1)

        # Преобразуем y_predict из (1, points_per_call) в (points_per_call,)
        if len(y_predict.shape) == 2 and y_predict.shape[0] == 1:
            y_predict = y_predict[0]  # Теперь y_predict имеет форму (points_per_call,)

        # Берем нужное количество точек
        y_predict = y_predict[:current_points_to_predict]

        # Добавляем предсказания в общий список
        # predict_values.extend(y_predict[:, 0])  # y_predict имеет размерность (current_points_to_predict, 1)
        predict_values.extend(y_predict)  # y_predict имеет размерность (current_points_to_predict, 1)


        # Обновляем x_input и x_future для следующих итераций
        for i in range(current_points_to_predict):
            privios_val = x_input[0, -1, 0]
            cur_val = y_predict[i]       # Текущий предсказанный элемент

            lcr = calc_lcr(previous_val=privios_val, cur_val=cur_val)  # Рассчитываем LCR

            x_input = np.delete(x_input, (0), axis=1)

            # Берем следующую lag из x_future
            future_lag = x_future[0]
            x_future = np.delete(x_future, 0, axis=0)

            # Обновляем lag текущим предсказанием
            future_lag[0] = cur_val
            x_input = np.append(x_input, future_lag.reshape(1, 1, -1), axis=1)

            # Обновляем LCR в последнем элементе
            x_input[0, -1, -1] = lcr

        # Уменьшаем количество оставшихся точек для предсказания
        remaining_horizon -= current_points_to_predict

    return predict_values

home_path = os.getcwd()


params_path = os.path.join(home_path, params_file)
params = yaml.load(open(params_path, 'r'), Loader=yaml.SafeLoader)
csv_train_data = params['csv_train_data']
lstm0_units = params['lstm0_units']
lstm1_units = params['lstm1_units']
lstm2_units = params['lstm2_units']

lag = params['lag']
activation = params['activation']
optimizer = params['optimizer']
dense_units = params['dense_units']
dropout_count = params['dropout_count']
epochs = params['epochs']
col_for_train = params['col_for_train']
points_per_call = params['points_per_call']

df_all_data = pd.read_csv(csv_train_data)
df_all_data['time'] = pd.to_datetime(df_all_data['time']).apply(lambda x: x.replace(second=0))
df_all_data = df_all_data.sort_values(by='time')
df_all_data['time'] = df_all_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_all_data = df_all_data.rename(columns={'load_consumption': 'P_l'})

tn = TimeNormalization(col_target='P_l', col_time='time')

df_all_data_norm, min_val, max_val = tn.df_normalize_with_meta(df_all_data)

# TODO Эта часть отвечает за подтягивание метапараметров ---------------------------------------------------------------
df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRzjnptk4SENCQOEH3cpi2MzpGlYen1v4b8xtE9ENs97_ObR0h2Kk7CZSZoGdNHy9PuVhSjYjTbQ_5I/pub?gid=419625624&single=true&output=csv')

df_all_data_norm[['temperature', 'pressure',
                  'dew_point', 'heat_index', 'humidity', 'solar_irradiance', 'uv_index',
                  'wind_chill']] = df[['temperature', 'pressure',
                                       'dew_point', 'heat_index', 'humidity', 'solar_irradiance', 'uv_index',
                                       'wind_chill']]

# TODO Дата с которой делаем прогноз на сутки вперед ------------------------------------------------------------------

# df_all_data_norm = df_all_data_norm.iloc[int(len(df_all_data_norm)/2):]
# df_all_data_norm = df_all_data_norm.iloc[:-20]

date_1 = date_for_test + ' 00:00:00'
date_2 = date_for_test + ' 00:05:00'
start_day = df_all_data[(df_all_data['time'] >= date_1) & (df_all_data['time'] <= date_2)]
start_day_index = start_day.index[0]
df_all_data_norm = df_all_data_norm[:start_day_index + 98 + 289]
all_col = df_all_data_norm.columns
print(f'Все доступные колонки - {all_col}')
diff_cols = all_col.difference(col_for_train)

# TODO Расчет LCR ------------------------------------------------------------------------------------------------------



df_all_data_norm['lcr'] = (df_all_data_norm['P_l'].shift(1) - df_all_data_norm['P_l']) / df_all_data_norm['P_l']
df_all_data_norm['lcr'] = df_all_data_norm['lcr'].shift(1)
df_all_data_norm = df_all_data_norm[2:]
df_all_data_norm = df_all_data_norm.reset_index()


# TODO Здесь создаем данные для обучение и прогноз ---------------------------------------------------------------------

# Данные На Тренировку
df = df_all_data_norm
train_index = len(df) - 288
df_train_all_col = df.loc[:train_index]
df_test_all_col = df.loc[train_index + 1:]
df_true_all_col = df_test_all_col.copy()
df = df_all_data_norm[col_for_train]
df_train = df.loc[:train_index]
values = df_train[col_for_train].values
# X, y = split_sequence(values, lag)

X, y = split_sequence(values, lag, points_per_call)

# Данные На предсказание
df_test = df.loc[train_index + 1:]
df_true = df_test.copy()
df_test['P_l'] = None
df_forecast = df_test.copy()
x_input = create_x_input(df_train, lag)
df_test_no_lcr = df_test.copy()
df_test_no_lcr['lcr'] = None
x_future = df_test_no_lcr.values
n_features = values.shape[1]


# TODO Здесь задаем конфигурацию модели слои и тд --------------------------------------------------------------------



save_best_weights_callback = SaveBestWeights()

model = Sequential()

# model.add(Bidirectional(LSTM(lstm0_units, activation=activation), input_shape=(lag, n_features)))

model.add(Bidirectional(LSTM(lstm0_units, activation='softplus', return_sequences=True), input_shape=(lag, n_features)))
# model.add(Dropout(0.1))

model.add(Bidirectional(LSTM(lstm1_units, activation=activation, return_sequences=True)))
# model.add(Dropout(0.1))

model.add(Bidirectional(LSTM(lstm2_units, activation=activation)))
# model.add(Dropout(0.1))
model.add(Dense(points_per_call, activation='linear',
                kernel_regularizer=regularizers.l2(0.005)))


model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# TODO Обучение --------------------------------------------------------------------------------------------------------
history = model.fit(X, y, epochs=epochs, verbose=1, callbacks=[save_best_weights_callback])

# TODO Прогноз ---------------------------------------------------------------------------------------------------------


x_input = create_x_input(df_train, lag)
x_input = x_input.reshape((1, lag, n_features))
predict_values = make_predictions_lcr_2(x_input, x_future, points_per_call)
# predict_values = make_predictions(x_input)

print(f"Прогнозируемые значения : {predict_values}")


predict_values = np.array(predict_values).flatten()

df_forecast['P_l'] = predict_values
df_forecast = replace_zeros_with_average(df_forecast, 'P_l')

if len(diff_cols) > 0:
    for col in diff_cols:
        df_forecast[col] = df_true_all_col[col]

df_forecast[col] = df_true_all_col[col]
df_comparative = tn.df_denormalize_with_meta(df_forecast, min_val, max_val)
df_true = tn.df_denormalize_with_meta(df_true_all_col, min_val, max_val)


# TODO Отрисовка -------------------------------------------------------------------------------------------------------

fig_p_l = make_subplots(rows=1, cols=1, subplot_titles=['P_l_real vs P_l_predict'])

fig_p_l.add_trace(
    go.Scatter(x=df_true['time'], y=df_true['P_l'], mode='lines', name='P_l_real', line=dict(color='blue')), row=1,
    col=1)
fig_p_l.add_trace(go.Scatter(x=df_comparative['time'], y=df_comparative['P_l'], mode='lines', name='P_l_predict',
                             line=dict(color='orange')), row=1, col=1)
template = "presentation"

fig_p_l.show()


y_true = df_true['P_l']
y_pred = df_comparative['P_l']

# TODO Метрики ---------------------------------------------------------------------------------------------------------

rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)
print(f'RMSE = {rmse}')
print(f'R-squared = {r2}')
print(f'MAE = {mae}')
print(f'MAPE = {mape}')
print(f'WMAPE = {wmape}')

model.summary()
# plot_model(model, show_shapes=True, to_file='model.png')
# fig_p_l.write_image("fig_p_l.png")


# TODO Сохранение модели -----------------------------------------------------------------------------------------------

name_model_dir = f'model_predict1'
model_dir_path = '/Users/nikitasavvin/Desktop/Учеба/LCR_forecast'
os.makedirs(model_dir_path, exist_ok=True)
model_name = f'{name_model_dir}.h5'
model.save(os.path.join(model_dir_path, model_name))

# params_file_path = os.path.join(model_dir_path, 'params.yaml')
# with open(params_file_path, 'w') as params_file:
#     yaml.dump(params, params_file, default_flow_style=False)

# TODO График потерь --------------------------------------------------------------------------------------------------

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)), y=history.history['loss'], mode='lines'))
# fig.update_layout(title='Model Training Loss (Interactive)',
#                   xaxis_title='Epoch',
#                   yaxis_title='Loss',
#                   template='plotly_dark',
#                   hovermode='x')
#
# html_file_path = os.path.join(model_dir_path, 'training_loss_plot_interactive.html')
# fig.show()
# fig.write_html(html_file_path)
# print(name_model_dir)
