import pandas as pd
# for SARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def buildModel():
    # Fit the seasonal ARIMA model
    # the original non-stationary dataset is used directly to fit the SARIMA model, since SARIMA can handle non-stationary data with seasonal patterns. 
    model = sm.tsa.statespace.SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res1 = model.fit()
    print('build - done')
    return res1

def test_set_results():
    start_test = test_data.index[0]
    end_test = test_data.index[len(test_data)-1]
   # Predicting values
    pred1 = res.predict(start = start_test, end = end_test)
    pred_df = pd.DataFrame(pred1)
    pred_df.index.name = 'date'
    pred1.to_csv('uploads/test_set_results.csv')
    print('Test set prediction - done')
    return pred1
   

def future_forecast(start,end):
    # Predicting values
    pred1 = res.predict(start = start, end = end)
    pred_df = pd.DataFrame(pred1)
    pred_df.index.name = 'date'
    pred1.to_csv('uploads/predictions.csv')
    print('prediction - done')
    return pred1

def metrics():
     # Evaluate the model
    mape = ((abs(df['Total'] - pred) / df['Total']).mean()) * 100
    print('MAPE:', mape)
    

def plotGraph(predicted):
    plt.plot(test_data['Total'], label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.title('Actual vs. Predicted Sales Data')
    plt.legend()
    plt.show()



def main(periodicity,num):
  # Reading the dataset 
 global df,pred,res,initial,final,dates,labels,train_data,test_data,test_results
 df = pd.read_csv('uploads/dataset.csv',index_col='Date', parse_dates=True)
 # Split the data into training and testing sets
 train_size = int(len(df) * 0.8)  # 80% for training
 train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
 res =  buildModel()
 test_results = test_set_results()
#  plotGraph(test_results)
  # Evaluation
#  metrics()
 if(periodicity=='Months'):
  
    # df['Date'] = pd.to_datetime(df['Date'])
    # print(df.head())

    # Difference the time series to make it stationary
    # df_diff = df.diff().dropna() #sales[i] = sales[i]-sales[i-1] , i = 2 to n
    # print(df_diff.head())
    initial = df.index[len(df)-1]
    final = initial + pd.Timedelta(days=31*(int(num)-1))
    pred = future_forecast(initial,final)

    # Plotting the graph
    # plotGraph()
 else:
    if(periodicity=='Days'): 
      # resample to daily frequency
      df = df.resample('D').sum()
    if(periodicity=='Weeks'):
       # resample to Weekly frequency
      df = df.resample('W').sum()
       

    # fit ETS model
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(df, seasonal_periods=12, trend='add', seasonal='add')
    model_fit = model.fit()

    initial = df.index[len(df)-1]
    if(periodicity=='Days'): final = initial + pd.Timedelta(days = int(num))
    if(periodicity=='Weeks'): final = initial + pd.Timedelta(weeks = int(num))
    print(initial)
    print(final)
    # make forecast
    forecast_daily = model_fit.predict(start=initial, end=final)
    pred = forecast_daily
    # print(forecast_daily)
    # plt.plot(forecast_daily)
    # plt.show()
    forecast_daily = pd.DataFrame(forecast_daily)
    forecast_daily.index.name = 'date'
    forecast_daily = forecast_daily.rename(columns={0: 'Predicted Sales'})
    forecast_daily.to_csv('uploads/predictions.csv')
    print('prediction - done')

# returns the data points for the graph
def datapts():
    # values of x axis
    dates = pred.index.values
    labels = []
    for x in dates:
      labels.append(str(x).split("T")[0])
    # print(labels)
    # Values of y -axis
    sales = pred.tolist()
    # print(sales)
    response = {
      "labels":labels,
      "sales":sales
     }
    # print(response)
    return response 


# returns the data points for test set
def datapts_test():
   dates = test_data.index.values
   labels = []
   for x in dates:
      labels.append(str(x).split("T")[0])
   sales = test_results.tolist()
   actual = test_data['Total'].tolist()
   response = {
      "labels":labels,
      "predicted":sales,
      "actual":actual
   }
   return response

# if __name__=="__main__":
#     main('Months',num=10)
#     print(datapts_test())
   
    



# # Step 8: Determine the parameters for the seasonal ARIMA model
# fig, ax = plt.subplots(2, 1, figsize=(12, 6))
# sm.graphics.tsa.plot_acf(df_diff, lags=12, ax=ax[0])
# sm.graphics.tsa.plot_pacf(df_diff, lags=12, ax=ax[1])
# plt.show()