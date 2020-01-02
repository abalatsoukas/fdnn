import itertools
import numpy as np
import scipy.io
from scipy.signal import savgol_filter, decimate
import sys
import matplotlib.pyplot as plt

# Loads testbed data from file
def loadData(fileName, params):

		# Get parameters
		dataOffset = params['dataOffset']
		chanLen = params['hSILen']
		offset = np.maximum(dataOffset-int(np.ceil(chanLen/2)),1)

		# Load the file
		matFile = scipy.io.loadmat(fileName)

		# Prepare data
		x = np.squeeze(matFile['txSamples'], axis=1)[:-offset]
		y = np.squeeze(matFile['analogResidual'], axis=1)[offset:]
		y = y - np.mean(y)
		noise = np.squeeze(matFile['noiseSamples'], axis=1)
		noisePower = np.squeeze(matFile['noisePower'], axis=1)

		# Return
		return x, y, noise, noisePower

# Plots PSD and computes various signal powers
def plotPSD(y_test, yCanc, yCancNonLin, noise, params, type, yVar=1):

	# Get self-interference channel length
	chanLen = params['hSILen']

	# Calculate signal powers
	if( type == 'NL' ):
		noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
		yTestPower = 10*np.log10(np.mean(np.abs(y_test[chanLen:])**2))
		yTestLinCancPower = 10*np.log10(np.mean(np.abs(y_test[chanLen:]-yCanc[chanLen:])**2))
		yTestNonLinCancPower = 10*np.log10(np.mean(np.abs(y_test[chanLen:]-yCancNonLin[chanLen:])**2))
	else:
		noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
		yTestPower = 10*np.log10(np.mean(np.abs(y_test)**2))
		yTestLinCancPower = 10*np.log10(np.mean(np.abs(y_test-yCanc)**2))
		yTestNonLinCancPower = 10*np.log10(np.mean(np.abs(((y_test-yCanc)/np.sqrt(yVar)-yCancNonLin)*np.sqrt(yVar))**2))

	# Calculate spectra
	samplingFreqMHz = params['samplingFreqMHz']
	fftpoints = 4096
	scalingConst = samplingFreqMHz*1e6
	freqAxis = np.linspace(-samplingFreqMHz/2,samplingFreqMHz/2,fftpoints)
	savgolWindow = 45
	savgolDegree = 1

	if( type == 'NL' ):
		noisefft = np.fft.fftshift(np.fft.fft(noise/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
		yTestFFT = np.fft.fftshift(np.fft.fft(y_test[chanLen:]/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
		yTestLinCancFFT = np.fft.fftshift(np.fft.fft((y_test[chanLen:]-yCanc[chanLen:])/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
		yTestNonLinCancFFT = np.fft.fftshift(np.fft.fft((y_test[chanLen:]-yCancNonLin[chanLen:])/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
	else:
		noisefft = np.fft.fftshift(np.fft.fft(noise/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
		yTestFFT = np.fft.fftshift(np.fft.fft(y_test/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
		yTestLinCancFFT = np.fft.fftshift(np.fft.fft((y_test-yCanc)/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
		yTestNonLinCancFFT = np.fft.fftshift(np.fft.fft((((y_test-yCanc)/np.sqrt(yVar)-yCancNonLin)*np.sqrt(yVar))/np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))

	# Plot spectra
	toPlotyTestFFT = 10*np.log10(savgol_filter(np.power(np.abs(yTestFFT),2),savgolWindow,savgolDegree))
	toPlotyTestLinCancFFT = 10*np.log10(savgol_filter(np.power(np.abs(yTestLinCancFFT),2),savgolWindow,savgolDegree))
	toPlotyTestNonLinCancFFT = 10*np.log10(savgol_filter(np.power(np.abs(yTestNonLinCancFFT),2),savgolWindow,savgolDegree))
	toPlotnoisefft = 10*np.log10(savgol_filter(np.power(np.abs(noisefft),2),savgolWindow,savgolDegree))

	plt.plot(freqAxis, toPlotyTestFFT, 'b-', freqAxis, toPlotyTestLinCancFFT, 'r-', freqAxis, toPlotyTestNonLinCancFFT, 'm-', freqAxis, toPlotnoisefft, 'k-')
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Power Spectral Density (dBm/Hz)')
	plt.title('Polynomial Non-Linear Cancellation')
	plt.xlim([ -samplingFreqMHz/2, samplingFreqMHz/2])
	plt.ylim([ -170, -90 ])
	plt.xticks(range(-int(samplingFreqMHz/2),int(samplingFreqMHz/2),2))
	plt.grid(which='major', alpha=0.25)
	plt.legend(['Received SI Signal ({:.1f} dBm)'.format(yTestPower), 'After Linear Digital Cancellation ({:.1f} dBm)'.format(yTestLinCancPower), 'After Non-Linear Digital Cancellation ({:.1f} dBm)'.format(yTestNonLinCancPower), 'Measured Noise Floor ({:.1f} dBm)'.format(noisePower)], loc='upper center')
	plt.savefig('figures/NL.pdf', bbox_inches='tight')
	plt.show()

	# Return signal powers
	return noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower

# Estimates parameters for linear cancellation
def SIestimationLinear(x, y, params):

	# Get channel length
	chanLen = params['hSILen']

	# Construct LS problem
	A = np.reshape([np.flip(x[i+1:i+chanLen+1],axis=0) for i in range(x.size-chanLen)], (x.size-chanLen, chanLen))

	# Solve LS problem
	h = np.linalg.lstsq(A, y[chanLen:])[0]

	# Output estimated channels
	return h

# Estimates parameters for non-linear cancellation
def SIestimationNonLinear(x, y, params):

	print("Self-interference channel estimation:")

	# Get PA non-linearity parameters
	pamaxorder = params.get('pamaxordercanc', 1)
	chanLen = params['hSILen']
	nBasisFunctions = int(( pamaxorder+1)/2*( (pamaxorder+1)/2 +1))

	# Apply PA non-linearities
	A = np.zeros((x.size-chanLen, nBasisFunctions*chanLen), dtype=np.complex128)

	matInd = 0
	for i in range(1,pamaxorder+1,2):
		for j in range(0,i+1):
			sys.stdout.write("\r1. Constructing basis functions... ({:d}/{:d})".format(int(matInd+1),nBasisFunctions))
			sys.stdout.flush()
			xnl = np.power(x,j)*np.power(np.conj(x),i-j)
			A[:,matInd*chanLen:(matInd+1)*chanLen] = np.reshape([np.flip(xnl[i+1:i+chanLen+1],axis=0) for i in range(xnl.size-chanLen)], (xnl.size-chanLen, chanLen))
			matInd += 1

	sys.stdout.write("\r1. Constructing basis functions... done!         \n")

	# Solve LS problem
	sys.stdout.write("2. Doing channel estimation... ")
	sys.stdout.flush()
	h = np.linalg.lstsq(A, y[chanLen:])[0]
	sys.stdout.write("done! Estimated a total of {:d} parameters.\n".format(h.size*2))

	# Output estimated channels
	return h

# Perform linear cancellation based on estimated parameters
def SIcancellationLinear(x, h, params):

	# Calculate the cancellation signal
	xcan = np.convolve(x, h, mode='full')
	xcan = xcan[0:x.size]

	# Output
	return xcan

# Perform non-linear cancellation based on estimated parameters
def SIcancellationNonLinear(x, h, params):

	# Get parameters
	pamaxorder = params['pamaxordercanc']
	chanLen = params['hSILen']
	nBasisFunctions = int(( pamaxorder+1)/2*( (pamaxorder+1)/2 +1))

	# Calculate the cancellation signal
	xnl = x
	xcan = np.zeros(x.size+chanLen-1, dtype=np.complex128)

	chanInd = 0
	for i in range(1,pamaxorder+1,2):
		for j in range(0,i+1):
			sys.stdout.write("\r1. Constructing basis functions and cancellation signal... ({:d}/{:d})".format(int(chanInd+1),int(nBasisFunctions)))
			sys.stdout.flush()
			xnl = np.power(x,j)*np.power(np.conj(x),i-j)
			xcan += np.convolve(xnl, h[chanInd*chanLen:(chanInd+1)*chanLen])
			chanInd += 1

	# Output
	xcan = xcan[0:x.size]
	return xcan
