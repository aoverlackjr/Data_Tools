# Collection of data analysis and system identification classes and functions. To be continually expanded.
import numpy as np
from math import exp, pi

# Gaussian kernel filter.
def GaussianKernel(data, kernel_size, sigma):

    # Get the size of the data
    nPoints = len(data)

    # define output list
    output = []

    # Run by all points
    for ptNr in range(nPoints):

        # Set sum of weights to zero and output data as well
        K_sum = 0.0
        out_data_pt = 0.0

        # Run along all points in the kernel window:
        for wPt in range(-kernel_size,kernel_size+1,1):

            # Get the data index:
            data_index = ptNr + wPt
            if ((data_index >= 0) and (data_index < nPoints)):
                K = exp( -(ptNr - wPt)**2  / (2.0*(sigma**2)))
                K_sum += K
                out_data_pt += K*data[data_index]


        # Divide the output by the sum of weights
        output.append(out_data_pt/K_sum)

    # Return the output
    return output

def extractFilteredProperties(data, filtered_data):
    # Function extracts some data from the original data and the
    # filtered data, in order to analyze statistically the
    # performance of the filter.
    # Assumes the data and filtered_data are arrays, and are of
    # equal length.

    # Check how big the data is.
    nPoints = len(data)

    # Pre-allocate the residual data.
    upper_lower_residual = [0.0]*nPoints
    residual             = [0.0]*nPoints

    for i in range(nPoints):
        residual[i] = data[i] - filtered_data[i]

    # Get standard deviation from the residual data.
    std_dev = np.std(residual)

    # Get the positive and negative residuals as normalized
    for i in range(nPoints):
        if residual[i] > 0:
            upper_lower_residual[i] = 1
        if residual[i] < 0:
            upper_lower_residual[i] = -1

    return std_dev, residual, upper_lower_residual

def getLagPeriod(upper_lower_residual):
   # Gets the length (in steps) of the leag lead period, on the
   # basis of the upper_lower residual vector.
   # Used for filter tuning.

   # Get size of input data.
   size = len(upper_lower_residual)

   # Set output to zero
   max_lag_period = 0

   prev_point = upper_lower_residual[0]
   accumulated_period = 0

   # Loop all points to find the maximum length of a period.
   for i in range(1,size):

       if upper_lower_residual[i] == prev_point:
           # If previous point is same as current one, add to
           # measured period length and move forward.
           accumulated_period = accumulated_period + 1
       else:
           # Apparently the period has come to an end.
           # Check if this period was bigger than the previous one:
           if accumulated_period > max_lag_period:
               # Store new maximum:
               max_lag_period = accumulated_period

           # Set accumulated period back to zero
           accumulated_period = 0

       prev_point = upper_lower_residual[i]

   return max_lag_period


def  findOptimalGaussianFilter(data, max_kernel, max_lag):
    # This function loops the filter with different sigma values,
    # until the lag properties are matched to the sigma values.
    # Heuristially, this is assumed to be a good filter, without
    # too much lag, but with good noise filtering.

    # Set to initial value:
    kernel_tuned = 1
    # Loop the length of the input space, just to cover the whole
    # ground. Should stop earlier though.
    lags = [0.0]*max_kernel
    for kernel_test in range(1,max_kernel+1):
        # By starting with sigma as 1, we start with a very noisy
        # first filter. (all-pass)
        # Try out the filter:
        output = GaussianKernel(data, kernel_test, max_kernel)
        # Get the properties with this output.
        std_dev_error, residuals, upper_lower_residual = extractFilteredProperties(data, output)
        # See if the lag period is about the same as the sigma.
        max_lag_period =  getLagPeriod(upper_lower_residual)
        lags[kernel_test] = max_lag_period
        # If the max lag is the same or bigger as the sigma value,
        # the testing is stopped.
        if max_lag_period > max_lag:
            kernel_tuned = kernel_test-1
            break

    return output, std_dev_error, kernel_tuned, lags


# Fast Fourier Transform tool to transform data to the frequency domain.
def fft(data):
    size = len(data)
    if size <= 1: return data
    even = fft(data[0::2])
    odd =  fft(data[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]
