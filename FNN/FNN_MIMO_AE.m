%==================================================================================%
%                                                                                  %
%                                  FNN_MIMO_AE.m                                   %
%                                                                                  %
%----------------------------------------------------------------------------------%
%                                                                                  %
%             Author: Songyan Xue                 Date: 27.10.2020                 %
%                                                                                  %
%         Development Environment: Matlab R2017a (9.2.0.556344)mac 64-bit          %
%                                                                                  %
%==================================================================================%
 
clear all;
clc;

%% ==================================== Parameters ====================================
global sys; 
global var;

sys.GPU = 0;

if sys.GPU == 1
    sys.tx = gpuArray(8);                                % number of transmit antenna
    sys.rx = gpuArray(8);                                % number of receive antenna
    sys.time_dof = sys.tx;                               % time-domain dof
    sys.code_rate = 1/sys.time_dof;                      % code rate
    sys.data_rate = gpuArray(2);                         % bit/codeword/user   
    sys.EbN0_training = 12 - 10*log10(1./sys.code_rate); % Eb/N0 
    sys.noise_var_training = 10^(sys.EbN0_training/20);
    sys.batch_size = gpuArray(100);
    sys.lr = gpuArray(0.001);                            % learning rate
    sys.epoch = gpuArray(10^8);                          % total training epoch
    sys.beta1 = gpuArray(0.9);                           % |
    sys.beta2 = gpuArray(0.999);                         % | Adam optimizer
    sys.eps = gpuArray(1e-8);                            % |
    sys.stack_size = gpuArray(100);                      % online BER calculator
    sys.channel_type = 'Rayleigh';
    
    sys.num_bi_LSTM_layer = gpuArray(2);                 % number of bi-LSTM layers
    sys.input_dimension = gpuArray(2*sys.rx);        % input dimension
    sys.hidden_dimension = gpuArray(500);                % hidden dimension
    sys.output_dimension = gpuArray(sys.data_rate);      % output dimension
    
    sys.Err = gpuArray(ones(1,sys.stack_size) * sys.batch_size * sys.data_rate * sys.tx);
    
elseif sys.GPU ==0 
    sys.tx = 4;                                          % number of transmit antenna
    sys.rx = 8;                                          % number of receive antenna
    sys.time_dof = sys.tx+1;                               % time-domain dof
    sys.code_rate = 1/sys.time_dof;                      % code rate
    sys.data_rate = 2;                                   % bit/codeword/user   
    sys.EbN0_training = 15 - 10*log10(1./sys.code_rate) - 10*log10(1./sys.data_rate); % Eb/N0 
    sys.noise_var_training = 10^(sys.EbN0_training/20);
    sys.batch_size = 1000;
    sys.lr = 0.001;                                      % learning rate
    sys.epoch = 10^8;                                    % total training epoch
    sys.beta1 = 0.9;                                     % |
    sys.beta2 = 0.999;                                   % | Adam optimizer
    sys.eps = 1e-8;                                      % |
    sys.stack_size = 100;                                % online BER calculator
    sys.channel_type = 'Rayleigh';
    
    sys.num_of_layer = 4;                                % number of bi-LSTM layers 
    sys.input_dimension = 2*sys.time_dof*sys.rx;         % input dimension
    sys.hidden_dimension = 1024;                         % hidden dimension
    sys.output_dimension = sys.data_rate*sys.tx;         % output dimension
    
    sys.Err = ones(1,sys.stack_size) * sys.batch_size * sys.data_rate * sys.tx;
else
    warning('GPU type undefined!')
end
%% =============================== Matrix Initilization ================================
parameter_initilization();

%% ================================= Constelation Set ===================================
constelation();
 
%% ================================== RNN Training Loop ======================================
for iteration = 1:sys.epoch
    sys.counter = 0;
    
    % Generate transmitted signal
    [information_bearing_bits,samples_com] = generate_signal();
    
    % Transmitter-side linear layer
    transmitted_signal = linear_layer(information_bearing_bits);
 
    % Generate channel
    generate_channel();
 
    % Received signal
    received_signal = [];
    for i = 1:sys.time_dof
        received_signal = [received_signal;var.H * transmitted_signal(:,:,i) + [real(var.noise(:,:,i));imag(var.noise(:,:,i))]];
    end
 
    % LSTM
    final_output = feedforward(received_signal);
      
    % Backpropagation
    gradient_rx = backpropagation_dense(final_output,samples_com,received_signal);          
 
    backpropagation_linear(gradient_rx,information_bearing_bits)
    
    % Adam
    adam(iteration);
    
    % Reset gradient
    reset_gradient();
    
    % Online BER
    if mod(iteration,sys.stack_size+1) ~= 0
        onlineBER(final_output,iteration);
    end
end