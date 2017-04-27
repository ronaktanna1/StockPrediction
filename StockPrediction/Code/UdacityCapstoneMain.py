from UdacityCapstoneFunctions import *
from UdacityCapstonePrintout import *


########## Set parameters ##############

# Most parameters are defined as lists so we can use loops to vary them

# Training period
train_start_date = '2009-01-01'
train_end_date = '2015-12-31'

# Test period
test_start_date = '2016-01-01'
test_end_date = '2016-06-31'

# List of companies used as features (from input file)
input_file = "Quandl_NASDAQ.csv"
company_df = pd.read_csv(input_file, names=['Ticker', 'Name'], comment='#')
t_list = company_df.Ticker.tolist()
ticker_list = [t_list]

# Choose the type of stock data to be used for preditions
sd_list = ["Open", "Close", "Variation", "High", "Low", "Split", "Ex-Dividend", "AdjOpen", "AdjClose", "AdjHigh", "AdjLow", "AdjVariation"]
stock_data_list = [sd_list]


# Choose how many previous days are used to make predictions
#nHist_list = [1,2,3,4,5,6,7,8,9]
nHist_list = [1]

# Choose a classification model
# Create classifiers
#model_list = [LogisticRegression(), GaussianNB(), LinearSVC(C=1.0),RandomForestClassifier(n_estimators=100)]
model_list = [LogisticRegression()]

# Choose whether NDX of previous days should be used as a feature
use_NDX = True

# Choose how the parameters are printed out in the output file
#print_mode = 'nHist' # Print nHist, train_score, cv_score, test_score
#print_mode = 'Model' # Print model, train_score, cv_score, test_score
#print_mode = 'Company' # Print company, train_score, cv_score, test_score
#print_mode = 'Stock' # Print stock_data, train_score, cv_score, test_score
#print_mode = 'General' # Print nCompanies, stock_data, nHist, tr_s, cv_s, te_s
#print_mode = 'ExtendedGeneral' # Print listOfComp, s_d, nHist, tr_s, cv_s, te_s
print_mode = False

########## Create and initialise the train and test datasets ##############

# Create train and test sets
train = dataset("Train", train_start_date, train_end_date)
test = dataset("Test", test_start_date, test_end_date)

# Load all companies data from a file
train.getAllCompanyData()
test.getAllCompanyData(train)

# Check if the companies requested as features are available in datasets
ticker_list = check_company_list(ticker_list, train, verbose=False)

# Create vectors to store accuracy scores
params = []
train_scores = []
cv_scores = []
test_scores = []

########## Preprocess data and create features ##########

for tickers, stock_data, nHist, model in [(tickers, stock_data, nHist, model) for tickers in ticker_list for stock_data in stock_data_list for nHist in nHist_list for model in model_list]:

	# Select which companies are used as features
	train.select_companies(tickers)
	test.select_companies(tickers)

	# Split datasets and create features from raw data
	train.create_features(stock_data, nHist, use_NDX=use_NDX)
	test.create_features(stock_data, nHist, use_NDX=use_NDX)

	# Make sure feature Matrix is not empty
	if len(train.features.columns == 0):
		print "Empty"

	# Print out feature matrix info 
        print_feature_matrix_info(train, print_features=False)
        print_feature_matrix_info(test, print_features=False)
	
########## Train classification model ##########

	# Train model
	model = train.train_model(model)
	
########## Get score for different datasets ##########

	# Use model to predict test data
	params.append(print_param(tickers,stock_data,nHist,model,print_mode))
	train_scores.append(train.train_score)
	cv_scores.append(train.cv_score)
	test_scores.append(test.predict_NDX(model))

if print_mode != False:
	out_file = open('Result_%s_Score_%s.csv' %(print_mode, train.start_year), 'w')
	result = zip(params, train_scores, cv_scores, test_scores)
	result.sort(key = lambda x: x[2], reverse=True)
	for a,b,c,d in result:
		out_file.write("%s, %.3f, %.3f, %.3f\n" %(a, b, c, d))
	out_file.close()
