from UdacityCapstoneFunctions import *

### Acquire Quandl data available at the time of the training period 
### Acquire NDX data 
### Save dataframes to files

# Get a list of companies and tickers
input_file = "Quandl_NASDAQ.csv"
df = pd.read_csv(input_file, names=['Ticker', 'Name'], comment='#')
ticker_list = df.Ticker.tolist()

# Define train and test sets
train = dataset("Train", '2009-01-01', '2015-12-31')
test = dataset("Test", '2016-01-01', '2016-12-31')

train.get_Quandl_data(ticker_list)
print train.effective_tick_list

test.get_Quandl_data(train.effective_tick_list)

# Get the NASDAQ 100 data (NDX)
train.get_NDX_data()
test.get_NDX_data()

train.df.to_csv('train_%s_%s.csv' %(train.start_year, train.end_year))
test.df.to_csv('test_%s_%s.csv' %(train.start_year, train.end_year))
