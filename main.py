import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

payment_data = pd.read_csv('data/paymentsFull.csv', delimiter=';')
print(payment_data.head())

payment_data.info()

payment_data = payment_data[['PAY_DATE','PAY','CNT']] # Extracting required columns
payment_data['PAY_DATE'] = pd.to_datetime(payment_data['PAY_DATE'].apply(lambda x: x.split()[0]),dayfirst=True) # Selecting only date
payment_data.set_index('PAY_DATE',drop=True,inplace=True) # Setting date column as index
print(payment_data.head())

fg, ax =plt.subplots(1,2,figsize=(20,7))
ax[0].plot(payment_data['PAY'],label='PAY',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Payment value',size=15)
ax[0].legend()

ax[1].plot(payment_data['CNT'],label='Count',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Count of payments',size=15)
ax[1].legend()

fg.show()

from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
payment_data[payment_data.columns] = MMS.fit_transform(payment_data)

print(payment_data.shape)

training_size = round(len(payment_data) * 0.80) # Selecting 80 % for training and 20 % for testing
print(training_size)

train_data = payment_data[:training_size]
test_data  = payment_data[training_size:]

print(train_data.shape, test_data.shape)

def create_sequence(dataset):
  sequences = []
  labels = []

  start_idx = 0

  for stop_idx in range(50,len(dataset)): # Selecting 50 rows at a time
    sequences.append(dataset.iloc[start_idx:stop_idx])
    labels.append(dataset.iloc[stop_idx])
    start_idx += 1
  return (np.array(sequences),np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

print(train_seq.shape, train_label.shape, test_seq.shape, test_label.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1))
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

model.fit(train_seq, train_label, epochs=40,validation_data=(test_seq, test_label), verbose=1)
# time.sleep(20)
test_predicted = model.predict(test_seq)
print(test_predicted[:5])

test_inverse_predicted = MMS.inverse_transform(test_predicted) # Inversing scaling on predicted data
print(test_inverse_predicted[:5])

gs_slic_data = pd.concat([payment_data.iloc[-699:].copy(),pd.DataFrame(test_inverse_predicted,columns=['payment_predicted','count_predicted'],index=payment_data.iloc[-699:].index)], axis=1)

gs_slic_data[['PAY','CNT']] = MMS.inverse_transform(gs_slic_data[['PAY','CNT']])

print(gs_slic_data.head())

gs_slic_data[['PAY','payment_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Payment Amount',size=15)
plt.title('Actual vs Predicted for payment amount',size=15)
plt.show()

gs_slic_data[['CNT','count_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Count amount',size=15)
plt.title('Actual vs Predicted for count amount',size=15)
plt.show()

gs_slic_data = gs_slic_data.append(pd.DataFrame(columns=gs_slic_data.columns,index=pd.date_range(start=gs_slic_data.index[-1], periods=11, freq='D', closed='right')))
print(gs_slic_data['2022-04-01	':'2022-04-3'])

upcoming_prediction = pd.DataFrame(columns=['PAY','CNT'],index=gs_slic_data.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)

curr_seq = test_seq[-1:]

for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)

upcoming_prediction[['PAY','CNT']] = MMS.inverse_transform(upcoming_prediction[['PAY','CNT']])

fg,ax=plt.subplots(figsize=(10,5))
ax.plot(gs_slic_data.loc['2021-04-01':,'PAY'],label='Current Payment Amount')
ax.plot(upcoming_prediction.loc['2021-04-01':,'PAY'],label='Upcoming Payment Amount')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Payment',size=15)
ax.set_title('Upcoming Payment amount prediction',size=15)
ax.legend()
fg.show()

fg,ax=plt.subplots(figsize=(10,5))
ax.plot(gs_slic_data.loc['2021-04-01':,'CNT'],label='Current count amount')
ax.plot(upcoming_prediction.loc['2021-04-01':,'CNT'],label='Upcoming count amount')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Count amount',size=15)
ax.set_title('Upcoming count amount prediction',size=15)
ax.legend()
fg.show()

plt.show()


