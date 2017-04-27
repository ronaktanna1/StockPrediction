# Date and time
import time
import datetime

# Data handling
import pandas as pd
import numpy as np

# Stock price data acquisition
import quandl
import pandas_datareader.data as web

# Machine learning
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import tree

# Ploting
import matplotlib.pyplot as plt

# Printing
from UdacityCapstonePrintout import *

# A Quandl key is required to access the Quandl database
# Quandl Key : MMRL4pygXxizDk5RozgX
quandl.ApiConfig.api_key = "MMRL4pygXxizDk5RozgX"

# Size of printed messages
print_size = 60

# Dataset Class: contain data about the train or test datasets 
# Data are stock prices for the companies of interest over a period of time
# Nasdaq index values for that period are also stored in the class
class dataset:
        def __init__(self, name = None, start = '0000-00-00', end = '0000_00_00'):
		# Name of the dataset
                self.name = name
		# Start date of the dataset
                self.start_date = start
		self.start_year = start.split('-')[0]
       		# End date of the dataset
	        self.end_date = end
		self.end_year = end.split('-')[0]
		# List of all companies
                self.cmpny_tick_list = []
		# Effective list of companies used to make predictions
                self.effective_tick_list = []
		# Pandas dataframe containing the raw data
                self.df = []
		# Pandas dataframe with the preprocessed data to be used 
		# as input for the predictions
		self.features = []
		# True values of NDX used to score the model performance
		self.target = []
		# Score on the train set (if applicable)
		self.train_score = 0.0
		# Score on the CV set (if applicable)
		self.cv_score = 0.0

	# Get dataframe with all the company data from a file
	def getAllCompanyData(self, train_dataset=None):

		# The companies available as features depend on 
		# the training period
		if (train_dataset == None):
			input_file = "train_%s_%s.csv" %(self.start_year,self.end_year)
		else:
			input_file = "%s_%s_%s.csv" %(self.name.lower(), train_dataset.start_year, train_dataset.end_year)
	
		self.ref_df = pd.read_csv(input_file)

	# Select companies used as features:
	def select_companies(self, list):
		self.cmpny_list = list # Desired companies
		drop_list = [] # Undesired companies
		for col in self.ref_df:
        		if (not (True in [col.find(x) >= 0 
			for x in list]) and (col.find("NDX") < 0)):
                		drop_list.append(col)
		self.df = self.ref_df.drop(drop_list, axis=1)

	# Get the data for each companies from the Quandl database
        def get_Quandl_data(self, list):
		print_acquisition_start(self)

		# If no Quandl data are used, create empty DataFrame
		if (len(list) == 0):
			self.df = pd.DataFrame({"A" : []})
			return 0
	
		# Get Quandl data for different comapnies
		# Store the data in a temporary list (data_list)
		# Merge the datasets at the end	
                data_list = []
                for tick in list:
                        tick = "WIKI/" + tick
			print tick
			data = quandl.get(tick, start_date=self.start_date, end_date=self.end_date)

			# Sanity check:
			# If company has no data at the desired start date
			# ignore this company 
			# May be it entered the stock market later
			if (int(self.start_year) != int(data.index[0].year) or int(self.start_date.split('-')[1]) != int(data.index[0].month)):
				print "Remove %s" %tick
				continue

			# Make a list of the companies effectively acquired
                        self.effective_tick_list.append(tick[5:])
                        data_list.append(data)

 			# Useful operations/cleaning on the data
 	                data_list[-1] = renameFeatures(data_list[-1])
        		data_list[-1] = calculate_variation(data_list[-1])
		print "\n"

		# Join all the data into one pandas DataFrame
                self.df = joinDataSets(data_list, self.effective_tick_list)

		# Fill the holes in the dataframe (forward fill)
                self.df.fillna(method='ffill', inplace=True)

		print_acquisition_end(self)

	# Get the values of the Nasdaq 100 index NDX
	def get_NDX_data(self):
		print_acquisitionNDX_start(self)

		# Use pandas built-in financial data reader to get NDX values
		data = web.DataReader("^NDX", 'yahoo', self.start_date, self.end_date) #['Adj Close']
		
		# Keep only the "AdjClose" data, the rest is not useful here
		data.drop([x for x in data.columns if x.find("Adj Close") < 0], axis=1, inplace=True)

		# Rename column
	        dict = {"Adj Close" : "NDX_AdjClose"}
        	data.rename(columns=dict, inplace = True)

		# Calculate the daily variation
		data["NDX_Variation"] = data["NDX_AdjClose"] - data["NDX_AdjClose"].shift(1)

		# Transform the absolute variation of AdjClose into 
		# "up" or "down" categorical variable
		data["NDX_UpDown"] = np.where(data["NDX_Variation"] >= 0.0, "Up", "Down")

		# Remove first row 
		# It contains a NaN in the Variation column
		data = data[1:]

                print_acquisitionNDX_end(data)

                print_QuandlNDX_merge_start(self)

		# Check that the length of the NDX data corresponds to the 
		# length of the Company data before concatenating datasets

		# If Quandl dataset is empty, df is just the NDX data
		if self.df.empty:
			self.df = data
			return 0

		# If Quandl data start date is earlier than NDX data, remove first Quandl rows
		while(data.index[0] > self.df.index[0]):
			self.df = self.df[1:]

		print_merge_info(self, data)

		# There might be days where NDX data are available 
		# but not Quandl
		# In that case we remove these days in the NDX dataset
		if (len(self.df) == len(data)):
			self.df = pd.concat([self.df, data], axis=1)
		else:
			# If the lengths don't match, drop the days in the NDX
			# data that are not present in the company dataset
			data.drop(data.index[[data.index.get_loc(x) for x in data.index if x not in self.df.index]], inplace=True)

			# Concatenate datasets
			self.df = pd.concat([self.df, data], axis=1)
			print "NDX data were truncated to match Quandl data"
	
	# Shift company data with respect to NDX data because we want to 
	# use past data to predict the future
	def shift_data(self, df, nHist, deltaT):
		print "Shift data to create historical features".center(print_size, '-')

	        ret_data = df
		feature_list = [x for x in df.columns]
        	for i in range (len(feature_list)):
        	 	for j in range (1, nHist + 1):
				shift = j + deltaT - 1
                        	histFeature = self.df[feature_list[i]].shift(shift)
                        	histFeature = histFeature.rename("%sT%d" %(feature_list[i],shift))
                        	ret_data = ret_data.join(histFeature)
			ret_data.drop([feature_list[i]], axis=1, inplace=True)
	        ret_data = ret_data[(nHist+deltaT-1):]
        	return ret_data

	def normalize_features(self):	
		scaler = preprocessing.MinMaxScaler()
		scaled = scaler.fit_transform(self.features)
		self.features = pd.DataFrame(scaled)
 
	def create_features(self, feat, nHist=1, deltaT = 1, use_NDX=True, normalize=False):
		create_feature_start(self)

		# The target column is NDX_UpDown
		target_name = "NDX_UpDown"

		self.target = self.df[target_name][(nHist+deltaT-1):]
		drop_list = [x for x in self.df.columns if not any("_%s" %y in x for y in feat)]
		if (use_NDX == False):
			drop_list.append("NDX_Variation")

		# Shift data to create historical features
		self.features = self.shift_data(self.df.drop(drop_list, axis=1), nHist, deltaT)

		# Normalize features
		if (normalize == True):
			self.normalize_features()

	def train_model(self, model):
		print "\n" + ("Dataset %s" %self.name).center(print_size, '=')
		print "Train model".center(print_size, '=')
		print "".center(print_size, '=')

		X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(self.features, self.target, train_size = 0.86, random_state=42)

	        trainModelStart = time.time()
		model.fit(X_train, Y_train)
       		print "%-56s %fs" %("Fit model in :", time.time() - trainModelStart)

        	scoreModelTrainStart = time.time()
        	self.train_score = model.score(X_train, Y_train)
        	print "%-40s %f Time : %fs" %("Score on train set :", self.train_score, time.time() - scoreModelTrainStart)
		scoreModelCVStart = time.time()
        	self.cv_score = model.score(X_valid, Y_valid)
        	print "%-40s %f Time : %fs" %("Score on cross validation set :", self.cv_score, time.time() - scoreModelCVStart)	

		return model

	def predict_NDX(self, model):
		self.predictNDX = pd.Series(model.predict(self.features), name="NDX_pred", index=self.target.index)
        	score = model.score(self.features, self.target)
                print "%-40s %f" %("Score on %s set : " %self.name, score)
		return score

# Functions
# Calculate the variation between market opening and closing
def calculate_variation(df):
        df["Variation"] = df["Close"].sub(df["Open"])
        df["AdjVariation"] = df["AdjClose"].sub(df["AdjOpen"])
        return df

# Change the name of feature so the company name appears
# (ex. AAPL_Open instead of Open)
def renameFeatures(df):
        dict = {"Adj. Close" : "AdjClose", "Adj. Open" : "AdjOpen", "Adj. High" : "AdjHigh", "Adj. Low" : "AdjLow", "Adj. Volume" : "AdjVolume"}
        df.rename(columns=dict, inplace = True)
        return df

# Join two pandas dataframes
def joinDataSets(dfList, symbolList):
        for i in range(len(dfList)):
                dfList[i].columns = ["%s_%s"%(symbolList[i], x) for x in dfList[i].columns]
        df = pd.concat([x for x in dfList], axis=1)
        return df

# Make sure all the companies requested as features are available
def check_company_list(ticker_list, dataset, verbose=False):
	return_list = []
	for list in ticker_list:
	        corrected_list = []
	        for c in list:
	                if (True in [x.find(c) >= 0 for x in dataset.ref_df.columns]):
	                        corrected_list.append(c)
	        if len(corrected_list) > 0:
	                return_list.append(corrected_list)

	return return_list
