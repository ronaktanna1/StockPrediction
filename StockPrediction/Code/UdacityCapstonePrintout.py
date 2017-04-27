### Print information to terminal or log file ###

# Printing parameters
# Size of printed messages
print_size = 60

def print_feature_matrix_info(dataset, print_features = False):

		print "\n"
		print ("Dataset : %s" %dataset.name).center(print_size, '=')

                print ("Feature Matrix size: %dx%d" %(len(dataset.features), len(dataset.features.columns))).center(print_size, ' ')
                print ("True value vector size: %d" %len(dataset.target)).center(print_size, ' ')
		
		if (print_features == True):
			s = ''
			for col in dataset.features.columns:
				s += "%s, " %col 
				if len(s) > 60:
					print s.center(print_size, ' ')	
					s=''
			print s.center(print_size, ' ')	

def print_param(tickers, stock_data, nHist, model, mode='General', ncluster=0):
	if mode == 'Tick':
		return "[%s]" %", ",join(tickers)
	elif mode == 'nHist':
		return "%d" %nHist 
	elif mode == 'Model':
		return "%s" %model 
	elif mode == 'Cluster':
		return "%d" %ncluster 
	elif mode == 'Stock':
		return "[%s]" %", ".join(stock_data)
	elif mode == 'ExtendedGeneral':
		return "[%s], stock data [%s], nHist %d" %(", ".join(tickers), ", ".join(stock_data), nHist)
	else:
		return "nCompanies %d, stock data [%s], nHist %d" %(len(tickers), ", ".join(stock_data), nHist)

def print_acquisition_start(dataset):
	print "\n" + ("Dataset %s" %dataset.name).center(print_size, '=')
        print "Start Acquisition of Quandl data".center(print_size, '=')
        print "".center(print_size, '=')

def print_acquisition_end(dataset):
        print ("Quandl Data: start date %s" %dataset.df.index[0]).center(print_size, ' ')
        print ("Quandl Data: size %d" %len(dataset.df)).center(print_size, ' ')
        print ("Quandl Data: nFeatures %d" %len(dataset.df.columns)).center(print_size, ' ')

def print_acquisitionNDX_start(dataset):
        print "\n" + ("Dataset %s" %dataset.name).center(print_size, '=')
        print "Start Acquisition of NDX data".center(print_size, '=')
        print "".center(print_size, '=')

def print_acquisitionNDX_end(data):
        print ("NDX Data: start date %s" %data.index[0]).center(print_size, ' ')
        print ("NDX Data: size %d" %len(data)).center(print_size, ' ')
	print ("NDX Data: nFeatures %d" %len(data.columns)).center(print_size, ' ')

def print_QuandlNDX_merge_start(dataset):
	print "\n" + ("Dataset %s" %dataset.name).center(print_size, '=')
	print "Start Merging of NDX  and Quandl data".center(print_size, '=')
        print "".center(print_size, '=')

def print_merge_info(data1, data2):
        print "Dataset contains Nans: %s" %data1.df.isnull().values.any()
        print "Start date %s " %data2.index[0]
        print "Length of NDX data %d" %len(data2)
        print "Length of Quandl data %d" %len(data1.df)

def create_feature_start(dataset):
        print "\n" + ("Dataset %s" %dataset.name).center(print_size, '=')
        print "Start creation of feature matrix".center(print_size, '=')
        print "".center(print_size, '=')
