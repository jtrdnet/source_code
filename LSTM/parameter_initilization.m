function parameter_initilization()
global sys;
global var;
    if sys.GPU == 1
            warning('GPU function undefined in the current version!')
    elseif sys.GPU == 0 
            var.linear_layer = (rand(2*sys.time_dof,2^sys.data_rate,1,sys.tx)-0.5)*0.02;
%             var.linear_layer = (1./sqrt(sys.time_dof) + (rand(2*sys.time_dof,2^sys.data_rate,1,sys.tx)-0.5)./sqrt(100*sys.time_dof)) .* sign(rand(2*sys.time_dof,2^sys.data_rate,1,sys.tx)-0.5);
            var.LSTM_weight_1 = (rand(sys.hidden_dimension,2*sys.rx,1,8)-0.5)*0.2;
            var.LSTM_weight_2 = (rand(sys.hidden_dimension,sys.hidden_dimension,1,8)-0.5)*0.2;
            var.LSTM_bias_b = zeros(sys.hidden_dimension,6,8+(sys.num_bi_LSTM_layer-1)*8+1);
            var.dense_weight = (rand(sys.data_rate,sys.hidden_dimension)-0.5)*0.2;
            var.dense_bias = zeros(sys.data_rate,5); 
            for i = 2:6
                var.linear_layer(:,:,i,:) = zeros(2*sys.time_dof,2^sys.data_rate,1,sys.tx);
                var.LSTM_weight_1(:,:,i,:) = zeros(sys.hidden_dimension,2*sys.rx,1,8);
%                 LSTM_weight_2(:,:,i,:) = zeros(sys.hidden_dimension,sys.hidden_dimension,1,8);
                var.dense_weight(:,:,i) = zeros(sys.data_rate,sys.hidden_dimension); 
            end
            if sys.num_bi_LSTM_layer == 1
                warning('Needs at least 2 bi-LSTM layers!')
            else
                for i = 1:(sys.num_bi_LSTM_layer-1)*16+1
                    var.LSTM_weight_2(:,:,1,8+i) = (rand(sys.hidden_dimension,sys.hidden_dimension)-0.5)*0.2;
                end
                for i = 2:6 
                    var.LSTM_weight_2(:,:,i,:) = zeros(sys.hidden_dimension,sys.hidden_dimension,1,8+(sys.num_bi_LSTM_layer-1)*16+1); 
                end  
            end
    else
        warning('GPU type undefined!')
    end

end