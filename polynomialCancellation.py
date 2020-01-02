import scipy.io
import numpy as np
import fullduplex as fd
import matplotlib.pyplot as plt

# Define system parameters
params = {
	'samplingFreqMHz': 20,	# Sampling frequency, required for correct scaling of PSD
	'hSILen': 13,			# Self-interference channel length
	'pamaxordercanc': 7,	# Maximum PA non-linearity order
	'trainingRatio': 0.9,	# Ratio of total samples to use for training
	'dataOffset': 14,		# Data offset to take transmitter-receiver misalignment into account
	}

##### Load and prepare data #####
x, y, noise, measuredNoisePower = fd.loadData('data/fdTestbedData'+str(params['samplingFreqMHz'])+'MHz10dBm', params)

# Get self-interference channel length
chanLen = params['hSILen']

# Print total number of real parameters to be estimated
print("Total number of real parameters to estimate for polynomial based canceller: {:d}".format(int(2 * chanLen * ((params['pamaxordercanc']+1)/2) * ((params['pamaxordercanc']+1)/2 + 1))))

# Split into training and test sets
trainingSamples = int(np.floor(x.size*params['trainingRatio']))
x_train = x[0:trainingSamples]
y_train = y[0:trainingSamples]
x_test = x[trainingSamples:]
y_test = y[trainingSamples:]

##### Training #####
# Estimate linear cancellation parameters
hLin = fd.SIestimationLinear(x_train, y_train, params)
# Estimate non-linear cancellation parameters
hNonLin = fd.SIestimationNonLinear(x_train, y_train, params)

##### Test #####
# Do linear cancellation only to get signal after linear cancellation for PSD plot
yCanc = fd.SIcancellationLinear(x_test, hLin, params)
# Do non-linear cancellation
yCancNonLin = fd.SIcancellationNonLinear(x_test, hNonLin, params)

##### Evaluation #####
# Scale signals according to known noise power
noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
scalingConst = np.power(10,-(measuredNoisePower-noisePower)/10)
noise /= np.sqrt(scalingConst)
y_test /= np.sqrt(scalingConst)
yCanc /= np.sqrt(scalingConst)
yCancNonLin /= np.sqrt(scalingConst)

# Plot PSD and get signal powers
noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower = fd.plotPSD(y_test, yCanc, yCancNonLin, noise, params, 'NL')

# Print cancellation performance
print('')
print('The linear SI cancellation is: {:.2f} dB'.format(yTestPower-yTestLinCancPower))
print('The non-linear SI cancellation is: {:.2f} dB'.format(yTestLinCancPower-yTestNonLinCancPower))
print('The noise floor is: {:.2f} dBm'.format(noisePower))
print('The distance from noise floor is: {:.2f} dB'.format(yTestNonLinCancPower-noisePower))
