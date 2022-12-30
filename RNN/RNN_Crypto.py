# Не доделан, не работает

import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import time
from sklearn import preprocessing  # pip установите sklearn... если у вас его нет!

SEQ_LEN = 60  # сколько времени предшествующей последовательности собирать для RNN
FUTURE_PERIOD_PREDICT = 3  # как далеко в будущее мы пытаемся предсказать?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def preprocess_df(df):
    df = df.drop(columns="future")  # это больше не нужно (колонка 'future').

    for col in df.columns:  # пройдите по всем столбцам
        if col != "target":  # нормализуйте все ... за исключением самой цели!
            df[col] = df[col].pct_change()  # изменение pct "нормализует" разные валюты (каждая криптомонета имеет значительно разные значения, нас действительно больше интересуют движения другой монеты)
            df.dropna(inplace=True)  # удалить nan, созданный с помощью pct_change
            df[col] = preprocessing.scale(df[col].values)  # масштаб от 0 до 1.

    df.dropna(inplace=True)  # снова очистка ... jic. Эти противные NAN любят подкрадываться.

    sequential_data = []  # это список, который будет СОДЕРЖАТЬ последовательности
    prev_days = deque(maxlen=SEQ_LEN)  # Это будут наши фактические последовательности. Они сделаны с помощью deque, который сохраняет максимальную длину, выталкивая старые значения по мере поступления новых

    for i in df.values:  # перебирать значения
        prev_days.append([n for n in i[:-1]])  # сохранить все, кроме целевого
        if len(prev_days) == SEQ_LEN:  # убедитесь, что у нас есть 60 последовательностей!
            sequential_data.append([np.array(prev_days), i[-1]])  # добавьте этих плохих парней!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # список, в котором будут храниться наши последовательности покупок и цели
    sells = []  # список, в котором будут храниться наши последовательности продаж и цели

    for seq, target in sequential_data:  # перебор последовательных данных
        if target == 0:  # если это "не покупка"
            sells.append([seq, target])  # добавить в список продаж
        elif target == 1:  # в противном случае, если целью является 1 ...
            buys.append([seq, target])  # это покупка!

    random.shuffle(buys)  # перетасуйте покупки
    random.shuffle(sells)  # перетасуйте продажи!

    lower = min(len(buys), len(sells))  # какая длина короче?

    buys = buys[:lower]  # убедитесь, что оба списка имеют только самую короткую длину.
    sells = sells[:lower]  # убедитесь, что оба списка имеют только самую короткую длину.

    sequential_data = buys+sells  # добавьте их вместе
    random.shuffle(sequential_data)  # еще один случай, чтобы модель не путалась со всеми 1 классом, а затем с другим.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


main_df = pd.DataFrame()  # Создаем пустой DF

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration

    ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
    print(ratio)
    dataset = f'training_datas/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)

## here, split away some slice of the future data from the main main_df.
times = sorted(main_df.index.values)  # получаем времена
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]   # получаем последние 5% времени

validation_main_df = main_df[(main_df.index >= last_5pct)]  # делаемданные проверки, в которых индекс находится в последних 5%
main_df = main_df[(main_df.index < last_5pct)]  # теперь main_df - это все данные до последних 5%

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.legacy.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))  # saves only the best ones

# Train model
# print(type(train_x), type(train_y))
train_y = np.array(train_y)  # Смена типа данных
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))
