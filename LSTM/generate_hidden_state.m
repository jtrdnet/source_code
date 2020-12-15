function generate_hidden_state()

global sys;
global var;

    if sys.GPU == 0
        var.hidden_output = zeros(sys.hidden_dimension,sys.batch_size,sys.time_dof+2,2,sys.num_bi_LSTM_layer);
        var.hidden_state = zeros(sys.hidden_dimension,sys.batch_size,sys.time_dof+2,2,sys.num_bi_LSTM_layer);
        var.gate_output = zeros(sys.hidden_dimension,sys.batch_size,sys.time_dof+1,4,2,sys.num_bi_LSTM_layer);
    elseif sys.GPU == 1
        var.hidden_output = gpuArray(zeros(sys.hidden_dimension,sys.batch_size,sys.time_dof+2,2,sys.num_bi_LSTM_layer));
        var.hidden_state = gpuArray(zeros(sys.hidden_dimension,sys.batch_size,sys.time_dof+2,2,sys.num_bi_LSTM_layer)); 
        var.gate_output = gpuArray(zeros(sys.hidden_dimension,sys.batch_size,sys.time_dof+1,4,2,sys.num_bi_LSTM_layer));
    else 
        warning('GPU type undefined!');
    end   
end

 