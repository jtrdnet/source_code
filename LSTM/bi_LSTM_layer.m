function output = bi_LSTM_layer(input,states)

global sys;
global var;
     
     if strncmpi(states,'input',5) == 1
         for i = 1:sys.time_dof   
            z_a_1 = var.LSTM_weight_1(:,:,1,1) * input(:,:,i) + var.LSTM_weight_2(:,:,1,1) * var.hidden_output(:,:,i,1,1) + kron(var.LSTM_bias_b(:,1,1),ones(1,sys.batch_size));
            var.gate_output(:,:,i,1,1,1) = tanh(z_a_1);

            z_i_1 = var.LSTM_weight_1(:,:,1,2) * input(:,:,i) + var.LSTM_weight_2(:,:,1,2) * var.hidden_output(:,:,i,1,1) + kron(var.LSTM_bias_b(:,1,2),ones(1,sys.batch_size));
            var.gate_output(:,:,i,2,1,1) = 1./(1+exp(-z_i_1));

            z_f_1 = var.LSTM_weight_1(:,:,1,3) * input(:,:,i) + var.LSTM_weight_2(:,:,1,3) * var.hidden_output(:,:,i,1,1) + kron(var.LSTM_bias_b(:,1,3),ones(1,sys.batch_size));
            var.gate_output(:,:,i,3,1,1) = 1./(1+exp(-z_f_1));

            z_o_1 = var.LSTM_weight_1(:,:,1,4) * input(:,:,i) + var.LSTM_weight_2(:,:,1,4) * var.hidden_output(:,:,i,1,1) + kron(var.LSTM_bias_b(:,1,4),ones(1,sys.batch_size));
            var.gate_output(:,:,i,4,1,1) = 1./(1+exp(-z_o_1));

            var.hidden_state(:,:,i+1,1,1) = var.gate_output(:,:,i,1,1,1) .* var.gate_output(:,:,i,2,1,1) + var.gate_output(:,:,i,3,1,1) .* var.hidden_state(:,:,i,1,1);       
            var.hidden_output(:,:,i+1,1,1) = tanh(var.hidden_state(:,:,i+1,1,1)) .* var.gate_output(:,:,i,4,1,1);
        end
     
        %% Backward
        for j = sys.time_dof+2:-1:3        
            z_a_2 = var.LSTM_weight_1(:,:,1,5) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,5) * var.hidden_output(:,:,j,2,1) + kron(var.LSTM_bias_b(:,1,5),ones(1,sys.batch_size));
            var.gate_output(:,:,j-1,1,2,1) = tanh(z_a_2);

            z_i_2 = var.LSTM_weight_1(:,:,1,6) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,6) * var.hidden_output(:,:,j,2,1) + kron(var.LSTM_bias_b(:,1,6),ones(1,sys.batch_size));
            var.gate_output(:,:,j-1,2,2,1) = 1./(1+exp(-z_i_2));

            z_f_2 = var.LSTM_weight_1(:,:,1,7) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,7) * var.hidden_output(:,:,j,2,1) + kron(var.LSTM_bias_b(:,1,7),ones(1,sys.batch_size));
            var.gate_output(:,:,j-1,3,2,1) = 1./(1+exp(-z_f_2));

            z_o_2 = var.LSTM_weight_1(:,:,1,8) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,8) * var.hidden_output(:,:,j,2,1) + kron(var.LSTM_bias_b(:,1,8),ones(1,sys.batch_size));
            var.gate_output(:,:,j-1,4,2,1) = 1./(1+exp(-z_o_2));

            var.hidden_state(:,:,j-1,2,1) = var.gate_output(:,:,j-1,1,2,1) .* var.gate_output(:,:,j-1,2,2,1) + var.gate_output(:,:,j-1,3,2,1) .* var.hidden_state(:,:,j,2,1);     
            var.hidden_output(:,:,j-1,2,1) = tanh(var.hidden_state(:,:,j-1,2,1)) .* var.gate_output(:,:,j-1,4,2,1);       
        end
         
        output =  var.hidden_output(:,:,2:sys.tx+1,1,1) + var.hidden_output(:,:,2:sys.tx+1,2,1);
        
     elseif strncmpi(states,'hidden',6) == 1
        for k = 1:(sys.num_bi_LSTM_layer-1)
            for i = 1:sys.time_dof   
                z_a_1 = var.LSTM_weight_2(:,:,1,16*k-7) * input(:,:,i) + var.LSTM_weight_2(:,:,1,16*k-3) * var.hidden_output(:,:,i,1,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+1),ones(1,sys.batch_size));
                var.gate_output(:,:,i,1,1,sys.num_bi_LSTM_layer) = tanh(z_a_1);

                z_i_1 = var.LSTM_weight_2(:,:,1,16*k-6) * input(:,:,i) + var.LSTM_weight_2(:,:,1,16*k-2) * var.hidden_output(:,:,i,1,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+2),ones(1,sys.batch_size));
                var.gate_output(:,:,i,2,1,sys.num_bi_LSTM_layer) = 1./(1+exp(-z_i_1));

                z_f_1 = var.LSTM_weight_2(:,:,1,16*k-5) * input(:,:,i) + var.LSTM_weight_2(:,:,1,16*k-1) * var.hidden_output(:,:,i,1,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+3),ones(1,sys.batch_size));
                var.gate_output(:,:,i,3,1,sys.num_bi_LSTM_layer) = 1./(1+exp(-z_f_1));

                z_o_1 = var.LSTM_weight_2(:,:,1,16*k-4) * input(:,:,i) + var.LSTM_weight_2(:,:,1,16*k) * var.hidden_output(:,:,i,1,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+4),ones(1,sys.batch_size));
                var.gate_output(:,:,i,4,1,sys.num_bi_LSTM_layer) = 1./(1+exp(-z_o_1));

                var.hidden_state(:,:,i+1,1,sys.num_bi_LSTM_layer) = var.gate_output(:,:,i,1,1,sys.num_bi_LSTM_layer) .* var.gate_output(:,:,i,2,1,sys.num_bi_LSTM_layer) + var.gate_output(:,:,i,3,1,sys.num_bi_LSTM_layer) .* var.hidden_state(:,:,i,1,sys.num_bi_LSTM_layer);       
                var.hidden_output(:,:,i+1,1,sys.num_bi_LSTM_layer) = tanh(var.hidden_state(:,:,i+1,1,sys.num_bi_LSTM_layer)) .* var.gate_output(:,:,i,4,1,sys.num_bi_LSTM_layer);
            end
            %% Backward
            for j = sys.time_dof+2:-1:3        
                z_a_2 = var.LSTM_weight_2(:,:,1,16*k+1) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,16*k+5) * var.hidden_output(:,:,j,2,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+5),ones(1,sys.batch_size));
                var.gate_output(:,:,j-1,1,2,sys.num_bi_LSTM_layer) = tanh(z_a_2);

                z_i_2 = var.LSTM_weight_2(:,:,1,16*k+2) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,16*k+6) * var.hidden_output(:,:,j,2,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+6),ones(1,sys.batch_size));
                var.gate_output(:,:,j-1,2,2,sys.num_bi_LSTM_layer) = 1./(1+exp(-z_i_2));

                z_f_2 = var.LSTM_weight_2(:,:,1,16*k+3) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,16*k+7) * var.hidden_output(:,:,j,2,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+7),ones(1,sys.batch_size));
                var.gate_output(:,:,j-1,3,2,sys.num_bi_LSTM_layer) = 1./(1+exp(-z_f_2));

                z_o_2 = var.LSTM_weight_2(:,:,1,16*k+4) * input(:,:,j-2) + var.LSTM_weight_2(:,:,1,16*k+8) * var.hidden_output(:,:,j,2,sys.num_bi_LSTM_layer) + kron(var.LSTM_bias_b(:,1,8*k+8),ones(1,sys.batch_size));
                var.gate_output(:,:,j-1,4,2,sys.num_bi_LSTM_layer) = 1./(1+exp(-z_o_2));

                var.hidden_state(:,:,j-1,2,sys.num_bi_LSTM_layer) = var.gate_output(:,:,j-1,1,2,sys.num_bi_LSTM_layer) .* var.gate_output(:,:,j-1,2,2,sys.num_bi_LSTM_layer) + var.gate_output(:,:,j-1,3,2,sys.num_bi_LSTM_layer) .* var.hidden_state(:,:,j,2,sys.num_bi_LSTM_layer);     
                var.hidden_output(:,:,j-1,2,sys.num_bi_LSTM_layer) = tanh(var.hidden_state(:,:,j-1,2,sys.num_bi_LSTM_layer)) .* var.gate_output(:,:,j-1,4,2,sys.num_bi_LSTM_layer);       
            end
            output =  var.hidden_output(:,:,2:sys.tx+1,1,sys.num_bi_LSTM_layer) + var.hidden_output(:,:,2:sys.tx+1,2,sys.num_bi_LSTM_layer);
            input = output;
        end 
     else
         warning('layer type undefined!')
     end
end