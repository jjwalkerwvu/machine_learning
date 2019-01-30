# as of 24 July, this works properly!
# There were some problems with comparing nx3 to nx1 arrays, and also the 
# labels of the irises were wrong.

# Import the necessary libraries
import numpy as np
# apparently we need to import random separately from np, because python
# is stupid
import numpy.random
# Need to import scipy.optimize, so that we will have Newton Conjugate 
# Gradient algorithm for doing our regression
#from scipy.optimize import minimize
# Need pandas, in order to use dataframes
import pandas
# need to import some way of reading the file, which, here, is .csv
import csv
# To run in python interpreter:
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
# Annoyed with putting my functions in separate .py files and figuring out
# how to get it to work that way, I will put them all here instead for the
# time being
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
def Sigmoid(z):
	return 1/(1 + np.exp(-z));
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
def log_grad_desc(X,y,theta,reg_param,alpha,num_iters,tol):
	# this function takes inputs:
	# NOTE: FOR NOW, ADDING THE EXTRA COLUMN IN X MATRIX AND THETA_0 WILL BE
	# DONE IN THE PARENT FUNCTION
	# theta 		= regression coefficients; add one element (regress. const.)
	# X					= the feature array for each training example. Assume that a
	#							column needs to be added, to correspond to regression const.
	# y					= the "actual" value for each training example
	# reg_param	=	the regularization parameter, usually much larger than 1
	#	alpha			= the learning parameter; higher number means faster learning
	# This function outputs:
	
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####

	# get m, number of training examples, and n, the number of features
	m,n=X.shape;
	# Not sure that these are needed, but included here anyway
	#y=y.reshape((m,1))
	#theta_coeffs=theta.reshape((n,1));
	theta_coeffs=theta

	# initialize cost function history
	J_hist= np.zeros((num_iters+1, 1))


#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
	# some initialization for the error, just make it larger than the tolerance by
	# some factor.
	err=tol*10
	# start off with 0 or 1?
	iter=0

	# the main loop goes here.
	while (iter<=num_iters-1) and (err>=tol):
		# compute the cost and gradient inside this function, just to reduce 
		# dependencies on other functions.

		# compute the cost? 
		term1 = np.log(Sigmoid(X.dot(theta_coeffs)));
 		term2 = np.log(1-Sigmoid(X.dot(theta_coeffs)));
		# the regularization term;
		# when calculating the squared norm of theta for this term, ignore theta[0],
		# the constant of regression (offset)
		term3 = reg_param/2/m*theta_coeffs[1:].T.dot(theta_coeffs[1:])
		term1 = term1.reshape((m,1))
 		term2 = term2.reshape((m,1))
 		term = y * term1 + (1 - y) * term2;
 		J_hist[iter] = -((np.sum(term))/m) + term3;
		##### grad=1/m*((sigmoid(X*theta)-y)'*X)'+lambda/m*[0;theta(2:end)];
		# compute grad here?
		sigmoid_x_theta = Sigmoid(X.dot(theta_coeffs));
		# the regularization term for the gradient:
		reg_term=reg_param/m*np.append(0,theta_coeffs[1:])
		# for some reason, reg_term turns into a row vector, and I can not 
		# transpose it back to a column vector
		reg_term=reg_term.reshape((len(reg_term),1))
		# do I need transpose of reg_term?
 		grad = ((X.T).dot(sigmoid_x_theta-y))/m+reg_term;
		# do I need to flatten? basically the same thing as grad(:) in matlab
		#grad=grad.flatten()

		# Now, we are ready to update the regression coefficients
		theta_coeffs=theta_coeffs-alpha*grad

		# compute the relative error:
		if iter>=1:
			err=(J_hist[iter-1]-J_hist[iter])/J_hist[iter]

		#print('The current loop number: %d ' % iter)
		#print('The current tolerance: %f ' % err)	
		#print('The cost function: %f ' % J_hist[iter])	
		
		iter=iter+1

	return [theta_coeffs,J_hist,iter]
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
# put the onevsall function here?
# y would be set equal to the current data set (train, cv, test)
# loop through all the cases
def onevsall(X,y,num_labels,reg_param):
	# m is the size of the data set, n is the number of features
	m,n=X.shape
	# remember, we have n+1 columns 
	all_theta=np.zeros((num_labels,n+1))
	# We must add a row of zeros to the feature matrix X; this corresponds to 
	# the theta_0 (offset, or constant)
	X=np.column_stack((np.ones([m,1]),X));
	
	# need to provide an initial theta for grad. desc., or some other algorithm;
	# just set the array to all zeros for simplicity here
	# use n+1, adding an extra for theta_0 (the offset, or constant in the 
	# regression)
	initial_theta=np.zeros([n+1,1])
	
	# I started with classes going from 1 to n, make sure not to get confused
	# with the ensuing indexing!
	for k in range(1,num_labels+1):
		# logical array for the kth class  
		y_logical=1.0*(y==k)
		# perform the logistic regression using grad. desc., or some other algorithm
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####		
		# calling logistic gradient descent algorithm
		# need a pretty small learning rate; I used 1e-4 here
		alpha=1e-1
		niters=1e3
		optimal_theta,J_hist,niters=log_grad_desc(X,y_logical,initial_theta,reg_param,alpha,niters,1e-5)
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####		
		# use Newton Conjugate Gradient algorithm to find the regression 
		# coefficients. Do I need to include the extra row of ones for this
		# algorithm?
		# as written, this uses truncated newton algorithm
		#result=minimize(fun=costfunc,x0=initial_theta,args = (X,y_logical),
  	#	method = 'TNC',jac = gradient)

		# is this line correct? I do not understand what it means, mostly the 
		# result.x part - looks like this is the solution array, the main thing
		# we want		
		#optimal_theta=result.x;
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
		# For the all_theta matrix, the first row corresponds to the regression
		# coefficients for the first class, the second row corresponds to 
		# regression coefficients for the second class, etc.
		all_theta[k-1,:]=optimal_theta.T
		#print(J_hist[1:niters])
		#print(k)
		print('Final value of cost function after grad. desc.: %f' % J_hist[niters-1])
		print('Number of iterations: %d' % niters)
	return all_theta
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
# START OF MAIN PROGRAM.
# this reading and cleaning of the data will be idiosyncratic to this  
# particular dataset, but adjust this as necessary for other datasets

filename='train_u6lujuX_CVtuZ9i.csv'

#with open('bezdekIris.data', 'rb') as csvfile:
#	iris_dimensions = csv.reader(csvfile, delimiter=' ')

# probably best to use the csv reader to import as a list?
#with open(filename, 'rb') as f:
#	reader = csv.reader(f)
# 	data = list(reader)

# Much better way! use pandas to make the dataframe.
data=pandas.read_csv(filename,encoding='utf-8')
# need to get all the names for the features/ class values of the data
# (the header)
header=list(data)

# find the number of rows and columns in our dataframe.
[rows,cols]=data.shape
# get the datatype for each
dt_col=[]
catvar_col=[]
# catvar_num goes with catvar_col list! it is the number of unique elements
# for each categorical variable
catvar_num=[]
num_catvars=0
for i in range(0,cols):
	# this list will tell you the data type for each column in the dataframe.
	dt=data[header[i]].dtype
	dt_col.append(str(dt))
	# if the data type is object, then get the number of unique elements in the
	# column; it is a categorical variable
	# I think 'object' is the only non-integer possibility to worry about?
	if 'object' in str(dt):
		catvar_col.append(i) 
		catvar_num.append(data[header[i]].nunique())
		# get an array with all of the categories for this independent variable
		data[header[i]].unique()
		# update the number of categorial variables counter
		num_catvars=num_catvars+1
		# my plan: if something is a categorical variable, then turn it into 
		# n-features, 0,1,2,...,n, with n being the number of categories
	# only an else is needed here; anything other than 'object' should be a
	# number, right? maybe boolean should be put into the if statement above...
	else
			
# turn catvar_col list into an np array?
#catvar_col=np.array(catvar_col);

# number of labels? get from the last column of the dataframe.
# assume the final column is the CLASS value.
num_labels=data[header[-1]].nunique()

# for the loan prediction dataset, the first column is just the loan id; not
# useful for running classification


	

#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
# Now that data has been imported in some sensible way, randomize the data,
# then set aside 40% for the training set, 40% for cross-validation set, and
# 20% for the test set.
# Note, sepal_length, sepal_width, petal_length, and petal_width are the
# features here.

# the random permutation of data_len number of indices:
# (gives a randomly ordered matrix from 0 to data_len-1, I think
rand_inds=numpy.random.permutation(data_len)
# comment the above line and Uncomment the following line if you do not want
# randomized data.
#rand_inds=range(1,data_len)
# before randomizing, save all of theses features and training examples
X_feat=np.column_stack((sepal_length,sepal_width,petal_length,petal_width))
y_class=iris_class
# now, randomize all of the data:
sepal_length=sepal_length[rand_inds]
sepal_width=sepal_width[rand_inds]
petal_length=petal_length[rand_inds]
petal_width=petal_width[rand_inds]
iris_class=iris_class[rand_inds]
print(iris_class)


# set up the indices
train_indices=[0,int(np.floor(0.4*data_len))]
cv_indices=[train_indices[1]+1,int(np.floor(0.8*data_len))]
test_indices=[cv_indices[1]+1,data_len-1]

# set up the training, cross-validation, and test sets
# training sets:
sl_train=sepal_length[train_indices[0]:train_indices[-1]]
sw_train=sepal_width[train_indices[0]:train_indices[-1]]
pl_train=petal_length[train_indices[0]:train_indices[-1]]
pw_train=petal_width[train_indices[0]:train_indices[-1]]
ic_train=iris_class[train_indices[0]:train_indices[-1]]
# make a feature matrix from all this data?
# annoyingly, python requires these double parenthesis, is this to
# reference a tuple?
X_train=np.column_stack((sl_train,sw_train,pl_train,pw_train))
# cv sets:
sl_cv=sepal_length[cv_indices[0]:cv_indices[-1]]
sw_cv=sepal_width[cv_indices[0]:cv_indices[-1]]
pl_cv=petal_length[cv_indices[0]:cv_indices[-1]]
pw_cv=petal_width[cv_indices[0]:cv_indices[-1]]
ic_cv=iris_class[cv_indices[0]:cv_indices[-1]]
# the feature matrix:
X_cv=np.column_stack((sl_cv,sw_cv,pl_cv,pw_cv))
# test sets:
sl_test=sepal_length[test_indices[0]:test_indices[-1]]
sw_test=sepal_width[test_indices[0]:test_indices[-1]]
pl_test=petal_length[test_indices[0]:test_indices[-1]]
pw_test=petal_width[test_indices[0]:test_indices[-1]]
ic_test=iris_class[test_indices[0]:test_indices[-1]]
# the feature matrix:
X_test=np.column_stack((sl_test,sw_test,pl_test,pw_test))
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
# should the features be normalized?
# where do I set up the one vs all? 
# (when testing for a specific case, the class in question gets value of 1,
# all others receive 0) 

# set reg_param=0 for now, still not sure how to put into code?
reg_param=0

# just looking at the training set for now:
all_theta=onevsall(X_train,ic_train,num_labels,reg_param)
## recall: X_feat is the enire dataset, not split into train, cv, and test 
## sets.
#all_theta=onevsall(X_feat,y_class,num_labels,reg_param)

# These definitions are made below in case you want to work on the entire set
# all at once instead of using the training set first.
#X_train=X_feat
#ic_train=y_class
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
# for now, let us calculate how well the regression coefficients predict the
# class inside of this script

# the prediction vector, which is m x 1
p = np.zeros((len(X_train),1))

# the number of labels we are dealing with
y_place=np.array(range(1,num_labels+1))

# have to add an extra column to which X matrix we are using, for the costant
# of regression
X_temp=np.column_stack((np.ones([len(X_train),1]),X_train))

temp1=Sigmoid(X_temp.dot(all_theta.T));
# amax is the maximum along a given axis of an array
max_arr=np.amax(temp1,axis=1)
# python "collapses" one of the dimensions when taking the array maximum, so
# we need to reshape in order to get mx1 instead of m,:
max_arr=max_arr.reshape(len(max_arr),1)
# I think this is ready to go.
# p is a matrix, where each column corresponds to a specific category
# (1 vs all, so 1 is the first class, 2 is the next class, etc.) AND
# each row is a specific trial or data example (m rows)
p=(max_arr*np.ones((1,num_labels))==temp1)*y_place
# Python cannot properly do the comparison with a nx3 to nx1 vector; so fix p
# so it has same dimensions as the iris set 
p=np.amax(p,axis=1)
p=p.reshape(len(p),1)


# display printout of how well the classifier works
#fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

# use ic_train for the moment, eventually make this a function?
# I think this use of mean works; the average is taken along the flattened 
# array by default, so the result tells us how well the algorithm worked 
# overall
accuracy=np.mean(1.0*(p==ic_train))*100

print('The accuracy of the logistic classifier: %f %%' % accuracy)
print(all_theta)
#print(p)
#print(J_hist[1:70])





