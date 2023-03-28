import pandas as pd
# for SARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt


def buildModel():
    # Fit the seasonal ARIMA model
    model = sm.tsa.statespace.SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res1 = model.fit()
    print('build - done')
    return res1

def predict(start,end):
    # Predicting values
    pred1 = res.predict(start = start, end = end)
    pred1.to_csv('uploads/predictions.csv')
    print('prediction - done')
    return pred1

def metrics():
     # Evaluate the model
    mape = ((abs(df['Total'] - pred) / df['Total']).mean()) * 100
    print('MAPE:', mape)

def plotGraph():
    plt.plot(df['Total'], label='Actual')
    plt.plot(pred, label='Predicted')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.title('Actual vs. Predicted Sales Data')
    plt.legend()
    plt.show()



def main(periodicity,num):
    global df,pred,res,initial,final
     # Reading the dataset 
    df = pd.read_csv('uploads/dataset.csv',index_col='Date', parse_dates=True)
    # df['Date'] = pd.to_datetime(df['Date'])
    print(df.head())

    # Difference the time series to make it stationary
    # df_diff = df.diff().dropna() #sales[i] = sales[i]-sales[i-1] , i = 2 to n
    # print(df_diff.head())
    
    res =  buildModel()
    # Evaluation
    # metrics()
    initial = df.index[len(df)-1]
    final = initial + pd.Timedelta(days=31*int(num))
    pred = predict(initial,final)

    # Plotting the graph
    plotGraph()

# returns the data points for the graph
def datapts():
    # values of x axis
    dates = pred.index.values
    labels = []
    for x in dates:
      labels.append(str(x).split("T")[0])
    
    # Values of y -axis
    sales = pred.tolist()
    response = {
      "labels":labels,
      "sales":sales
     }
    return response 


if __name__=="__main__":
    main(1,num=36)
   
    



# # Step 8: Determine the parameters for the seasonal ARIMA model
# fig, ax = plt.subplots(2, 1, figsize=(12, 6))
# sm.graphics.tsa.plot_acf(df_diff, lags=12, ax=ax[0])
# sm.graphics.tsa.plot_pacf(df_diff, lags=12, ax=ax[1])
# plt.show()