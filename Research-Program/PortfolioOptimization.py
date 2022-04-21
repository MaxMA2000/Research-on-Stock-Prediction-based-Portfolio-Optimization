"""
This program is the same functions in 
step5-stock_portfolio_optimization_models.ipynb
Run this program can directly builds up all 
portfolio optimization models and output results
"""


def get_stock_code_company_name(stock_code):
    """
    This function returns the corresponding company name
    of a certain stock code
    """
    import pandas as pd
    all_stock_data = pd.read_csv('../DataSource/research_use_39_stocks.csv')
    stock_code_df = all_stock_data[all_stock_data['Stock Code'] == stock_code]
    company_name = stock_code_df.iloc[0]["Company Name"]
    return company_name

def get_selected_stocks_df(stock_code_list, start_date, end_date):
    """
    This function returns a dataframe containing all the stocks
    needed in one portfolio with their historical prices
    """
    import pandas as pd
    from functools import reduce
    
    stock_df_list = []
    
    print("Start extracting all stocks ......")
    
    for stock_code in stock_code_list:
        data_file_location = '../DataSource/StockData/' + stock_code + ".csv"
        df = pd.read_csv(data_file_location)[["date", "close"]]
        df = df[(df['date']>=start_date) & (df['date']<=end_date)]
        
        company_name = get_stock_code_company_name(stock_code)
        stock_name = stock_code + "-" + company_name
        
        df = df.rename(columns={"close": stock_name})
        stock_df_list.append(df)
        
    df_merged = reduce(lambda left,right: pd.merge(left,right,on=['date'],how='outer'), stock_df_list)
    
    df_merged = df_merged.dropna()
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    df_merged = df_merged.set_index('date')
    print(f"Finish outputing {len(stock_code_list)} stocks: {[str(i) for i in stock_code_list]}" )
    return df_merged


def add_portfolio_return(data, portfolio_name_list, portfolio_weight_list):
    """
    This function adds the portfolio return based on 
    the weight of each stock and their historical price
    """
    import numpy as np
    import pandas as pd
    sub_data = data.copy()
    sub_data.loc[:, portfolio_name_list] *= np.array(portfolio_weight_list)
    data['portfolio_return'] = sub_data[portfolio_name_list].sum(axis=1)
    return data


def plot_portfolio_with_stocks(df, stock_list, period_name, model_name, prediction_model):
    """
    This function plots all the stocks price history 
    with the new weighted portfolio price history
    """
    from matplotlib import pyplot as plt
    from matplotlib import style
    import matplotlib.ticker as ticker

    # Plot the original Stock Price and Model Prediction
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(18,8))

    ax.plot(df, label=df.columns)

    # Adjust the labels and ticks for the plot
    title = f"{period_name}\n - {model_name}\n- {prediction_model} Prediction"

    xlabel = f"Stock Date Period: {str(df.index[0]).split(' ')[0]} - {str(df.index[-1]).split(' ')[0]}"
    ylabel = "Close Stock Price (HKD)"

    ax.set_title(title,fontsize=25, pad=20)
    ax.set_xlabel(xlabel, fontsize=18,fontstyle='oblique', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=18,fontstyle='oblique', labelpad=15)
    ax.legend(fontsize=14, facecolor='white')

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    figure_file_name = f"[{period_name}]-[{model_name}]-[{prediction_model}]]"
    figure_file_location = "../Results/PortfolioResultFigures/" + figure_file_name + ".png"
    plt.savefig(figure_file_location, bbox_inches='tight')
    print("\nSave figure in the " + figure_file_location)



def equal_weight_portfolio(stock_list, start_date, end_date, period_name, prediction_model):
    """
    This function builds up a equal-weighted portfolio based on stock list and the date period,
    and it returns the portfolio weight, annual sharpe ratio, cumulative return of the portfolio
    """
    import pandas as pd
    import numpy as np
    # Get the stock portfolios
    df = get_selected_stocks_df(stock_list, start_date, end_date)
    # Calculate the equal weight
    equal_weight = 1 / len(df.columns)
    weight = dict({stock:equal_weight for stock in df.columns})
    print("Portfolio Weight:" + str(weight))

    # Calculate the Sharpe ratio
    portfolio_name, portfolio_weight = list(weight.keys()), list(weight.values())
    log_return = np.sum(np.log(df/df.shift())*portfolio_weight, axis=1)
    sharpe_ratio = log_return.mean()/log_return.std()

    annual_sharpe_ratio = sharpe_ratio*252**.5
    print("Portfolio Sharpe Ratio: " + str(annual_sharpe_ratio))
    
    df = add_portfolio_return(df, portfolio_name, portfolio_weight)

    start_date_portfolio_value, end_date_portfolio_value = df['portfolio_return'][0], df['portfolio_return'][-1]
    
    plot_portfolio_with_stocks(df, stock_list, period_name, "Equal Weighted Optimization", prediction_model)
    
    return weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, end_date_portfolio_value-start_date_portfolio_value


def mean_variance_portfolio(stock_list, start_date, end_date, model_name, period_name, prediction_model):
    """
    This function builds up a Mean-Variance Optimized portfolio based on stock list and the date period,
    and it returns the portfolio weight, annual sharpe ratio, cumulative return of the portfolio
    """
    import pandas as pd
    df = get_selected_stocks_df(stock_list, start_date, end_date)

    # Estimate the expected returns and covariance matrix from the historical data
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage

    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()

    # Mean-Variance Optimization: find the portfolio that maximises the Sharpe Ratio
    from pypfopt.efficient_frontier import EfficientFrontier

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("Portfolio Weight:" + str(dict(cleaned_weights)))

    # Print out the expected performance of the portfolio
    annual_sharpe_ratio = ef.portfolio_performance(verbose=True)[-1]

    weight = dict(cleaned_weights)

    portfolio_name, portfolio_weight = list(weight.keys()), list(weight.values())
    
    """
    Functions to plot the efficient frontier with Monte Carlo Simulation
    """
    # from pypfopt import CLA, plotting

    # n_samples = 50000
    # w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    # rets = w.doat(mu)
    # stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
    # sharpes = rets / stds

    # print("Sample portfolio returns:", rets)
    # print("Sample portfolio volatilities:", stds)

    # # Plot efficient frontier with Monte Carlo sim
    # ef = EfficientFrontier(mu, S)

    # fig, ax = plt.subplots(figsize=(10,6))
    # plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # # Find and plot the tangency portfolio
    # ef2 = EfficientFrontier(mu, S)
    # ef2.max_sharpe()
    # ret_tangent, std_tangent, _ = ef2.portfolio_performance()

    # # Plot random portfolios
    # ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # cla = CLA(mu, S)
    # cla.max_sharpe()
    # cla.portfolio_performance(verbose=True);
    # ax = plotting.plot_efficient_frontier(cla, showfig=False)

    # # Format
    # ax.set_title("Efficient Frontier with random portfolios for \nAll Time Period-LSTM Network Prediction", fontsize=17)
    # ax.legend(fontsize=12)
    # plt.show()

    df = add_portfolio_return(df, portfolio_name, portfolio_weight)

    start_date_portfolio_value, end_date_portfolio_value = df['portfolio_return'][0], df['portfolio_return'][-1]
    
    plot_portfolio_with_stocks(df, stock_list, period_name, model_name, prediction_model)
    return weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, end_date_portfolio_value-start_date_portfolio_value


def hierarchical_risk_parity_portfolio(stock_list, start_date, end_date, period_name, prediction_model):
    """
    This function builds up a Hierarchical Risk Parity optimized portfolio 
    based on stock list and the date period,
    and it returns the portfolio weight, annual sharpe ratio, cumulative return of the portfolio
    """
    df = get_selected_stocks_df(stock_list, start_date, end_date)

    # Import the packages and build up portfolio
    from pypfopt import expected_returns

    rets = expected_returns.returns_from_prices(df)

    from pypfopt import HRPOpt
    hrp = HRPOpt(rets)
    hrp.optimize()
    weights = hrp.clean_weights()

    # from pypfopt import plotting
    # plotting.plot_dendrogram(hrp);

    # Print out the expected performance of the portfolio
    annual_sharpe_ratio = hrp.portfolio_performance(verbose=True)[-1]

    weight = dict(weights)

    portfolio_name, portfolio_weight = list(weight.keys()), list(weight.values())

    df = add_portfolio_return(df, portfolio_name, portfolio_weight)

    start_date_portfolio_value, end_date_portfolio_value = df['portfolio_return'][0], df['portfolio_return'][-1]
    
    plot_portfolio_with_stocks(df, stock_list, period_name, 'Hierarchical Risk Parity Optimization', prediction_model)
    
    return weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, end_date_portfolio_value-start_date_portfolio_value


def kmean_mean_variance(stock_list, start_date, end_date, period_name, prediction_model): 
    """
    This function builds up a K-Mean Clustering based Mean-Variance optimized portfolio 
    based on stock list and the date period,
    and it returns the portfolio weight, annual sharpe ratio, cumulative return of the portfolio
    """
    # Import the packages from scikit-learn and the Mean-Variance Optimization function
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    df = get_selected_stocks_df(stock_list, start_date, end_date)
    
    # Calculate the returns and variances for the portfolio
    daily_returns = df.pct_change()
    annual_mean_returns = daily_returns.mean() * 222
    annual_return_variance = daily_returns.var() * 222

    df2 = pd.DataFrame(df.columns, columns=['stock_symbols'])

    df2['returns'] = annual_mean_returns.values
    df2['variances'] = annual_return_variance.values

    # Use the Silquoutte Score method to select k - number of clusters
    from sklearn.metrics import silhouette_samples, silhouette_score

    # Build up K-Mean clusters based on returns and variances of each stock
    X = df2[['returns', 'variances']].values

    silhouette_avg = []
    for i in range(2, len(X)-1):
        kmeans_fit = KMeans(n_clusters = i).fit(X)
        silhouette_avg.append(silhouette_score(X, kmeans_fit.labels_))

    best_perform_num = silhouette_avg.index(max(silhouette_avg)) + 2

    kmeans = KMeans(n_clusters = best_perform_num).fit(X)
    labels = kmeans.labels_

    df2['cluster_labels'] = labels

    ## Plot the K-Mean Cluster plot
    # plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
    # plt.title('K-Means Plot')
    # plt.xlabel('Returns')
    # plt.ylabel('Variances')
    # plt.show()

    cluster_group = df2['cluster_labels'].value_counts().idxmax()
    df3 = df2[df2['cluster_labels'] == cluster_group]
    chosen_stocks = df3['stock_symbols'].tolist()

    # Re-select the dataframe with same group, samiliar level of return-risk
    df = df[chosen_stocks]

    # Run the Mean-Variance Optimization function for the updated portfolio dataframe
    weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return = mean_variance_portfolio(stock_list, start_date, end_date, "K-Mean based Mean-Variance Optimization", period_name, prediction_model)

    return weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return



def get_all_portfolio_optimization_results_per_stock_list(time_period_results, prediction_model, stock_list):
    """
    This function builds up all four portfolio optimization models for one given portfolio
    and return a dataframe containing the time period, prediction model for the portfolio,
    as well as the optimal weight, sharpe ratio and cumulative return
    """
    import pandas as pd
    pd.set_option('display.max_colwidth', None)

    # Give the period time based on user input period name
    if time_period_results == 'all_time_results': 
        period_time = ('2012-01-01', '2022-01-01')
        period_name = 'All Time Period'
    elif time_period_results == 'pre_covid_time_results':
        period_time = ('2012-01-01', '2020-01-09')
        period_name = 'Pre Covid Time Period'
    elif time_period_results == 'covid_time_results':
        period_time = ('2020-01-09', '2022-01-01')
        period_name = 'Covid Time Period'
    elif time_period_results == 'pre_covid_test_time_results':
        period_time = ('2018-01-09', '2020-01-01')
        period_name = 'Pre Covid Test Time Period'

    result = pd.DataFrame(columns=['time_period', 'prediction_model', 'portfolio_input', 'portfolio_optimization_model',
                                   'model_weight','sharpe', 'start', 'end', 'cumulative_return'])

    (start_date, end_date) = period_time
    
    portfolio_optimization_model_list = []

    # Equal Weight Portfolio
    result_input_list = [time_period_results, prediction_model, stock_list, 'Equal Weight Portfolio']
    weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return = equal_weight_portfolio(stock_list, start_date, end_date, period_name, prediction_model)
    result_input_list.extend([weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return])
    result.loc[len(result)] = result_input_list
    # display(result)

    # Mean Variance Portfolio
    result_input_list = [time_period_results, prediction_model, stock_list, 'Mean Variance Portfolio']
    weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return = mean_variance_portfolio(stock_list, start_date, end_date, "Mean-Variance Optimization", period_name, prediction_model)
    result_input_list.extend([weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return])
    result.loc[len(result)] = result_input_list

    # Hierarchical Risk Parity Portfolio
    result_input_list = [time_period_results, prediction_model, stock_list, 'Hierarchical Risk Parity Portfolio']
    weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return = hierarchical_risk_parity_portfolio(stock_list, start_date, end_date, period_name, prediction_model)
    result_input_list.extend([weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return])
    result.loc[len(result)] = result_input_list


    # K-Mean Clustering based Mean-Variance Optimization
    result_input_list = [time_period_results, prediction_model, stock_list, 'K-Mean Clustering based Mean-Variance Optimization']
    weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return = kmean_mean_variance(stock_list, start_date, end_date, period_name, prediction_model)
    result_input_list.extend([weight, annual_sharpe_ratio, start_date_portfolio_value, end_date_portfolio_value, cumulative_return])
    result.loc[len(result)] = result_input_list

    return result



def get_all_portfolio_input_for_all_optimization_results(portfolio_input_df):
    """
    This function uses the get_all_portfolio_optimization_results_per_stock_list function
    to run through all the portfolios in the portfolio_input_df dataframe
    and output a dataframe containing all portfolio with all optimization algorithms,
    4 periods x 4 prediction algorithms x 4 portfolio optimization algorithms = 64 results
    """
    final_result_list = []
    
    for num in range(len(portfolio_input_df)):
        # Collect time_period_results, prediction_model, and stock_list for each row of portfolio_input_df
        time_period_results = portfolio_input.loc[num, 'time period']
        prediction_model = portfolio_input.loc[num, 'model']
        stock_list = portfolio_input.loc[num, 'portfolio stock input']
        stock_list = stock_list.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        stock_list = list(stock_list.split(","))
        
        # Run the four portfolio optimization and save results
        row_result = get_all_portfolio_optimization_results_per_stock_list(time_period_results, prediction_model, stock_list)
        final_result_list.append(row_result)
        
    final_result_df = pd.concat([df for df in final_result_list], axis=0)

    final_result_df = final_result_df.reset_index(drop=True)
    final_result_df['portfolio_input'] = final_result_df['portfolio_input'].astype(str)
    return final_result_df


# Run through all the portfolios and store the value inside csv & xlsx file

import pandas as pd
portfolio_input = pd.read_csv('../Results/StockPrediction/portfolio_input_all_period_top5.csv')

result = get_all_portfolio_input_for_all_optimization_results(portfolio_input)

result.to_csv('../Results/PortfolioOptimization/portfolio_optimization_results_all_period_prediction.csv')
result.to_excel('../Results/PortfolioOptimization/portfolio_optimization_results_all_period_prediction.xlsx')


def get_HSI_increase_percentage_to_result():
    """
    This function reads the previous stored final results, and add the 
    Hang Seng Index increase percentage. It also calculates the increase percentage
    of the optimal portfolio for comparision
    """
    df = pd.read_csv('../Results/PortfolioOptimization/portfolio_optimization_results_all_period_prediction.csv')
    df = df.drop(columns='Unnamed: 0')
    df['cumulative_increase_percentage(%)'] = df['cumulative_return'] / df['start'] * 100
    
    # Get the HSI data
    HSI = pd.read_csv('../DataSource/StockData/HK.800000.csv', usecols=['date', 'close'])

    # Define the function to calculate increase percentage
    def get_hsi_certain_period_increase_percentage(df):
        df_begin, df_end = df.loc[0, 'close'], df.loc[len(df)-1, 'close']
        increase_percentage = ((df_end - df_begin) / df_begin) * 100
        return increase_percentage
    
    # Get different time period of HSI to add to the final portfolio results
    HSI_all = HSI[(HSI['date']>='2012-01-01') & (HSI['date']<='2022-01-01')].copy().reset_index(drop=True)
    HSI_covid = HSI[(HSI['date']>='2020-01-09') & (HSI['date']<='2022-01-01')].copy().reset_index(drop=True)
    HSI_pre_covid = HSI[(HSI['date']>='2012-01-01') & (HSI['date']<='2020-01-09')].copy().reset_index(drop=True)
    HSI_pre_covid_test = HSI[(HSI['date']>='2018-01-09') & (HSI['date']<='2020-01-01')].copy().reset_index(drop=True)

    # Calculate the increase percentage of HSI during certain periods
    all_result = get_hsi_certain_period_increase_percentage(HSI_all)
    covid_result = get_hsi_certain_period_increase_percentage(HSI_covid)
    pre_covid_result = get_hsi_certain_period_increase_percentage(HSI_pre_covid)
    pre_covid_test_result = get_hsi_certain_period_increase_percentage(HSI_pre_covid_test)

    # Create a dataframe to store the increase percentage of HSI during different periods
    HSI_df_increase = pd.DataFrame(columns=['time_period', 'HSI_increase_percentage(%)'])

    HSI_df_increase.loc[len(HSI_df_increase.index)] = ['all_time_results', all_result]
    HSI_df_increase.loc[len(HSI_df_increase.index)] = ['covid_time_results', covid_result]
    HSI_df_increase.loc[len(HSI_df_increase.index)] = ['pre_covid_time_results', pre_covid_result]
    HSI_df_increase.loc[len(HSI_df_increase.index)] = ['pre_covid_test_time_results', pre_covid_test_result]
    
    # Merge the HSI to the portfolio final results
    df = df.merge(HSI_df_increase, left_on='time_period', right_on='time_period', how='left')
    df = df.set_index(['time_period', 'prediction_model', 'portfolio_input'])
    
    return df



# Finalized the final results with HSI added for comparision
final_result = get_HSI_increase_percentage_to_result()

final_result.to_csv('../Results/PortfolioOptimization/portfolio_optimization_results_all_period_prediction.csv')
final_result.to_excel('../Results/PortfolioOptimization/portfolio_optimization_results_all_period_prediction.xlsx')
