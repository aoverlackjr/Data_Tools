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





'''

c

        function state_transition_matrix = transition_matrix(data, dt)

            # Asses size of data. Data should be inserted as column data,
            # where each row is equally spaced in time.
            [nSteps,nVar] = size(data);

            # Get the time derivatives of the data:
            var_dot = SIT.differentiate(data,dt);

            # Pre-allocate the aggregate transition matrix.
            TM = zeros(nVar*(nSteps-1),nVar*nVar);

            # Pre allocate the aggregate state matrix.
            V = zeros(nVar*(nSteps-1),1);

            # Fill the aggregate matrix and vector.
            for i = 1:nVar
                row = (i-1)*nSteps + 1;
                col = 1 + (i-1)*nVar;
                TM(row:row+nSteps-2,col:col+nVar-1) = data(1:end-1,:);
                V(row:row+nSteps-2,1) = var_dot(:,i);
            end

            # Perform pseudo inverse to obtain the state transition matrix in
            # vector form.
            pseudo_inverse = (TM'*TM)\TM';
            vectorMatrix = pseudo_inverse*V;

            # Refactor the state transition matrix from the vector
            state_transition_matrix = zeros(nVar,nVar);
            for r = 1:nVar
                base = 1+(r-1)*nVar;
                state_transition_matrix(r,:) = vectorMatrix(base:base+nVar-1,1)';
            end

            # The matrix is still in the form x_1 = x_0 + A*x_0;
            # Adding the unity matrix will make it of the form: x_1 =
            # A*x_0;
            state_transition_matrix = state_transition_matrix + eye(nVar);

        end

        function var_dot = differentiate(data,dt)

            [nSteps,nVar] = size(data);

            var_dot = zeros(nSteps-1,nVar);

            # Calculate the first derivative.
            for vr = 1:nVar
                for step = 1:nSteps-1
                    var_dot(step,vr) = ( data(step+1,vr) - data(step,vr) ) / dt;
                end
            end

        end

        function [state_matrices_history, bias_matrix, std_matrix] = examine_state_matrix_evolution(data,window, dt)
            # Runs along the data, including more every time, and generates
            # an evolution of TM matrices, to see if it converges over
            # time, an as such if the matrix is converging to an inherent
            # behaviour.

            # Get data size:
            [nPoints,nVars] = size(data);

            # If window is set to -1 the window is incrementally increased.
            if window == -1

                # Pre allocate results
                state_matrices_history = zeros(nPoints,nVars,nVars);

                # First data set must include at least a data length of nVars,
                # otherwise the pseudo inverse does not work
                index = 1;
                for end_pt = nVars:nPoints
                    input_data = data(1:end_pt,:);
                    state_matrices_history(index,:,:) = SIT.transition_matrix(input_data, dt);
                    index = index + 1;
                end

            else

                end_point = nPoints-window+1;
                state_matrices_history = zeros(end_point,nVars,nVars);
                ind = 1;
                for start_index = 1:end_point
                    input_data = data(start_index:start_index+window-1,:);
                    state_matrices_history(ind,:,:) = SIT.transition_matrix(input_data, dt);
                    ind = ind + 1;
                end


            end

            # Get the bias of the elements (mean) as wel as the std
            # deviation.
            bias_matrix = zeros(nVars,nVars);
            std_matrix  = zeros(nVars,nVars);
            for i = 1:nVars
                for j = 1:nVars
                    bias_matrix(i,j) = mean(state_matrices_history(:,i,j));
                    std_matrix(i,j) = std(state_matrices_history(:,i,j));
                end
            end


        end


        end





    end

end

'''
