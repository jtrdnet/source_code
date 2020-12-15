function gradient_hat = backpropagation_LSTM(gradient,input,a)

global sys;
global var;
 
 if strncmpi(a,'LSTM_1',6) ~= 1
     if sys.GPU == 0
        k = str2double(a(isstrprop(a,'digit'))) - 1;
        diff_backward_output = zeros(size(gradient(:,:,1)));
        diff_forward_output = zeros(size(gradient(:,:,1)));

        diff_output_1 = zeros(size(input));
        diff_output_2 = zeros(size(input));
     elseif sys.GPU == 1
        k = gpuArray(str2double(a(isstrprop(a,'digit'))) - 1);
        diff_backward_output = gpuArray(zeros(size(gradient(:,:,1))));
        diff_forward_output = gpuArray(zeros(size(gradient(:,:,1))));

        diff_output_1 = gpuArray(zeros(size(input)));
        diff_output_2 = gpuArray(zeros(size(input))); 
     end
    for i = 1 : sys.tx   
        diff_backward_output_2_t = gradient(:,:,i) + diff_backward_output;
        if i == 1
            diff_backward_output_2_state = diff_backward_output_2_t .* var.gate_output(:,:,i+1,1,2,sys.num_bi_LSTM_layer) .* (1 - tanh(var.hidden_state(:,:,i+1,2,sys.num_bi_LSTM_layer)).^2); 
        else
            diff_backward_output_2_state = diff_backward_output_2_t .* var.gate_output(:,:,i+1,4,2,sys.num_bi_LSTM_layer) .* (1 - tanh(var.hidden_state(:,:,i+1,2,sys.num_bi_LSTM_layer)).^2) + diff_backward_output_2_state .* var.gate_output(:,:,i,3,2,sys.num_bi_LSTM_layer);            
        end

        diff_a_1 = diff_backward_output_2_state .* var.gate_output(:,:,i+1,2,2,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,i+1,1,2,sys.num_bi_LSTM_layer).^2);
        diff_i_1 = diff_backward_output_2_state .* var.gate_output(:,:,i+1,1,2,sys.num_bi_LSTM_layer) .* var.gate_output(:,:,i+1,2,2,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,i+1,2,2,sys.num_bi_LSTM_layer));
        diff_f_1 = diff_backward_output_2_state .* var.hidden_state(:,:,i+2,2,sys.num_bi_LSTM_layer) .* var.gate_output(:,:,i+1,3,2,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,i+1,3,2,sys.num_bi_LSTM_layer));
        diff_o_1 = diff_backward_output_2_t .* tanh(var.hidden_state(:,:,i+1,2,sys.num_bi_LSTM_layer)) .* var.gate_output(:,:,i+1,4,2,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,i+1,4,2,sys.num_bi_LSTM_layer));


        var.LSTM_weight_2(:,:,2,16*k+1) = var.LSTM_weight_2(:,:,2,16*k+1) + diff_a_1 * input(:,:,i).';
        var.LSTM_weight_2(:,:,2,16*k+2) = var.LSTM_weight_2(:,:,2,16*k+2) + diff_i_1 * input(:,:,i).';
        var.LSTM_weight_2(:,:,2,16*k+3) = var.LSTM_weight_2(:,:,2,16*k+3) + diff_f_1 * input(:,:,i).';
        var.LSTM_weight_2(:,:,2,16*k+4) = var.LSTM_weight_2(:,:,2,16*k+4) + diff_o_1 * input(:,:,i).';

        var.LSTM_weight_2(:,:,2,16*k+5) = var.LSTM_weight_2(:,:,2,16*k+5) + diff_a_1 * var.hidden_output(:,:,i+2,2,sys.num_bi_LSTM_layer).';
        var.LSTM_weight_2(:,:,2,16*k+6) = var.LSTM_weight_2(:,:,2,16*k+6) + diff_i_1 * var.hidden_output(:,:,i+2,2,sys.num_bi_LSTM_layer).';
        var.LSTM_weight_2(:,:,2,16*k+7) = var.LSTM_weight_2(:,:,2,16*k+7) + diff_f_1 * var.hidden_output(:,:,i+2,2,sys.num_bi_LSTM_layer).';
        var.LSTM_weight_2(:,:,2,16*k+8) = var.LSTM_weight_2(:,:,2,16*k+8) + diff_o_1 * var.hidden_output(:,:,i+2,2,sys.num_bi_LSTM_layer).';

        var.LSTM_bias_b(:,2,8*k+5) = var.LSTM_bias_b(:,2,8*k+5) + sum(diff_a_1,2);
        var.LSTM_bias_b(:,2,8*k+6) = var.LSTM_bias_b(:,2,8*k+6) + sum(diff_i_1,2);
        var.LSTM_bias_b(:,2,8*k+7) = var.LSTM_bias_b(:,2,8*k+7) + sum(diff_f_1,2);
        var.LSTM_bias_b(:,2,8*k+8) = var.LSTM_bias_b(:,2,8*k+8) + sum(diff_o_1,2);

        diff_output_1(:,:,i) = var.LSTM_weight_2(:,:,1,16*k+1).' * diff_a_1 + var.LSTM_weight_2(:,:,1,16*k+2).' * diff_i_1 + var.LSTM_weight_2(:,:,1,16*k+3).' * diff_f_1 + var.LSTM_weight_2(:,:,1,16*k+4).' * diff_o_1;
        diff_backward_output = var.LSTM_weight_2(:,:,1,16*k+5).' * diff_a_1 + var.LSTM_weight_2(:,:,1,16*k+6).' * diff_i_1 + var.LSTM_weight_2(:,:,1,16*k+7).' * diff_f_1 + var.LSTM_weight_2(:,:,1,16*k+8).' * diff_o_1;

        diff_forward_output_2_t = gradient(:,:,sys.tx+1-i) + diff_forward_output;
        if i == 1
            diff_forward_output_2_state = diff_forward_output_2_t .* var.gate_output(:,:,sys.tx+1-i,1,1,sys.num_bi_LSTM_layer) .* (1 - tanh(var.hidden_state(:,:,sys.tx+2-i,1,sys.num_bi_LSTM_layer)).^2);
        else
            diff_forward_output_2_state = diff_forward_output_2_t .* var.gate_output(:,:,sys.tx+1-i,4,1,sys.num_bi_LSTM_layer) .* (1 - tanh(var.hidden_state(:,:,sys.tx+2-i,1,sys.num_bi_LSTM_layer)).^2) + diff_forward_output_2_state .* var.gate_output(:,:,sys.tx+2-i,3,1,sys.num_bi_LSTM_layer);            
        end

        diff_a_2 = diff_forward_output_2_state .* var.gate_output(:,:,sys.tx+1-i,2,1,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,sys.tx+1-i,1,1,sys.num_bi_LSTM_layer).^2);
        diff_i_2 = diff_forward_output_2_state .* var.gate_output(:,:,sys.tx+1-i,1,1,sys.num_bi_LSTM_layer) .* var.gate_output(:,:,sys.tx+1-i,2,1,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,sys.tx+1-i,2,1,sys.num_bi_LSTM_layer));
        diff_f_2 = diff_forward_output_2_state .* var.hidden_state(:,:,sys.tx+1-i,1,sys.num_bi_LSTM_layer) .* var.gate_output(:,:,sys.tx+1-i,3,1,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,sys.tx+1-i,3,1,sys.num_bi_LSTM_layer));
        diff_o_2 = diff_forward_output_2_t .* tanh(var.hidden_state(:,:,sys.tx+2-i,1,sys.num_bi_LSTM_layer)) .* var.gate_output(:,:,sys.tx+1-i,4,1,sys.num_bi_LSTM_layer) .* (1 - var.gate_output(:,:,sys.tx+1-i,4,1,sys.num_bi_LSTM_layer));

        var.LSTM_weight_2(:,:,2,16*k-7) = var.LSTM_weight_2(:,:,2,16*k-7) + diff_a_2 * input(:,:,sys.tx+1-i).';
        var.LSTM_weight_2(:,:,2,16*k-6) = var.LSTM_weight_2(:,:,2,16*k-6) + diff_i_2 * input(:,:,sys.tx+1-i).';
        var.LSTM_weight_2(:,:,2,16*k-5) = var.LSTM_weight_2(:,:,2,16*k-5) + diff_f_2 * input(:,:,sys.tx+1-i).';
        var.LSTM_weight_2(:,:,2,16*k-4) = var.LSTM_weight_2(:,:,2,16*k-4) + diff_o_2 * input(:,:,sys.tx+1-i).';

        var.LSTM_weight_2(:,:,2,16*k-3) = var.LSTM_weight_2(:,:,2,16*k-3) + diff_a_2 * var.hidden_output(:,:,sys.tx+1-i,1,sys.num_bi_LSTM_layer).';
        var.LSTM_weight_2(:,:,2,16*k-2) = var.LSTM_weight_2(:,:,2,16*k-2) + diff_i_2 * var.hidden_output(:,:,sys.tx+1-i,1,sys.num_bi_LSTM_layer).';
        var.LSTM_weight_2(:,:,2,16*k-1) = var.LSTM_weight_2(:,:,2,16*k-1) + diff_f_2 * var.hidden_output(:,:,sys.tx+1-i,1,sys.num_bi_LSTM_layer).';
        var.LSTM_weight_2(:,:,2,16*k) = var.LSTM_weight_2(:,:,2,16*k) + diff_o_2 * var.hidden_output(:,:,sys.tx+1-i,1,sys.num_bi_LSTM_layer).';

        var.LSTM_bias_b(:,2,8*k+4) = var.LSTM_bias_b(:,2,8*k+4) + sum(diff_a_2,2);
        var.LSTM_bias_b(:,2,8*k+3) = var.LSTM_bias_b(:,2,8*k+3) + sum(diff_i_2,2);
        var.LSTM_bias_b(:,2,8*k+2) = var.LSTM_bias_b(:,2,8*k+2) + sum(diff_f_2,2);
        var.LSTM_bias_b(:,2,8*k+1) = var.LSTM_bias_b(:,2,8*k+1) + sum(diff_o_2,2);

        diff_output_2(:,:,sys.tx+1-i) = var.LSTM_weight_2(:,:,1,16*k-7).' * diff_a_2 + var.LSTM_weight_2(:,:,1,16*k-6).' * diff_i_2 + var.LSTM_weight_2(:,:,1,16*k-5).' * diff_f_2 + var.LSTM_weight_2(:,:,1,16*k-4).' * diff_o_2;
        diff_forward_output = var.LSTM_weight_2(:,:,1,16*k-3).' * diff_a_2 + var.LSTM_weight_2(:,:,1,16*k-2).' * diff_i_2 + var.LSTM_weight_2(:,:,1,16*k-1).' * diff_f_2 + var.LSTM_weight_2(:,:,1,16*k).' * diff_o_2;
    end
    gradient_hat = (diff_output_1 + diff_output_2)./2;
    
 elseif strncmpi(a,'LSTM_1',6) == 1
     if sys.GPU == 0 
        diff_backward_output_1_h = zeros(size(gradient(:,:,1)));
        diff_forward_output_1_h = zeros(size(gradient(:,:,1)));
        
        diff_output_1 = zeros(size(input));
        diff_output_2 = zeros(size(input));
     elseif sys.GPU == 1 
        diff_backward_output_1_h = gpuArray(zeros(size(gradient(:,:,1))));
        diff_forward_output_1_h = gpuArray(zeros(size(gradient(:,:,1))));
     end
 
    for i = 1 : sys.tx    
        diff_backward_output_1_t = gradient(:,:,i) + diff_backward_output_1_h;
        if i == 1
            diff_backward_output_1_state = diff_backward_output_1_t .* var.gate_output(:,:,i+1,4,2,1) .* (1 - tanh(var.hidden_state(:,:,i+1,2,1)).^2); 
        else
            diff_backward_output_1_state = diff_backward_output_1_t .* var.gate_output(:,:,i+1,4,2,1) .* (1 - tanh(var.hidden_state(:,:,i+1,2,1)).^2) + diff_backward_output_1_state .* var.gate_output(:,:,i,3,2,1);            
        end
        
        diff_a_2 = diff_backward_output_1_state .* var.gate_output(:,:,i+1,2,2,1) .* (1 - var.gate_output(:,:,i+1,1,2,1).^2);
        diff_i_2 = diff_backward_output_1_state .* var.gate_output(:,:,i+1,1,2,1) .* var.gate_output(:,:,i+1,2,2,1) .* (1 - var.gate_output(:,:,i+1,2,2,1));
        diff_f_2 = diff_backward_output_1_state .* var.hidden_state(:,:,i+2,2,1) .* var.gate_output(:,:,i+1,3,2,1) .* (1 - var.gate_output(:,:,i+1,3,2,1));
        diff_o_2 = diff_backward_output_1_t .* tanh(var.hidden_state(:,:,i+1,2,1)) .* var.gate_output(:,:,i+1,4,2,1) .* (1 - var.gate_output(:,:,i+1,4,2,1));
    
        var.LSTM_weight_1(:,:,2,5) = var.LSTM_weight_1(:,:,2,5) + diff_a_2 * input(:,:,i).';
        var.LSTM_weight_1(:,:,2,6) = var.LSTM_weight_1(:,:,2,6) + diff_i_2 * input(:,:,i).';
        var.LSTM_weight_1(:,:,2,7) = var.LSTM_weight_1(:,:,2,7) + diff_f_2 * input(:,:,i).';
        var.LSTM_weight_1(:,:,2,8) = var.LSTM_weight_1(:,:,2,8) + diff_o_2 * input(:,:,i).';
        
        var.LSTM_weight_2(:,:,2,5) = var.LSTM_weight_2(:,:,2,5) + diff_a_2 * var.hidden_output(:,:,i+2,2,1).';
        var.LSTM_weight_2(:,:,2,6) = var.LSTM_weight_2(:,:,2,6) + diff_i_2 * var.hidden_output(:,:,i+2,2,1).';
        var.LSTM_weight_2(:,:,2,7) = var.LSTM_weight_2(:,:,2,7) + diff_f_2 * var.hidden_output(:,:,i+2,2,1).';
        var.LSTM_weight_2(:,:,2,8) = var.LSTM_weight_2(:,:,2,8) + diff_o_2 * var.hidden_output(:,:,i+2,2,1).';
        
        var.LSTM_bias_b(:,2,5) = var.LSTM_bias_b(:,2,5) + sum(diff_a_2,2);
        var.LSTM_bias_b(:,2,6) = var.LSTM_bias_b(:,2,6) + sum(diff_i_2,2);
        var.LSTM_bias_b(:,2,7) = var.LSTM_bias_b(:,2,7) + sum(diff_f_2,2);
        var.LSTM_bias_b(:,2,8) = var.LSTM_bias_b(:,2,8) + sum(diff_o_2,2);
        
        diff_output_1(:,:,i) = var.LSTM_weight_1(:,:,1,5).' * diff_a_2 + var.LSTM_weight_1(:,:,1,6).' * diff_i_2 + var.LSTM_weight_1(:,:,1,7).' * diff_f_2 + var.LSTM_weight_1(:,:,1,8).' * diff_o_2;
        diff_backward_output_1_h = var.LSTM_weight_2(:,:,1,5).' * diff_a_2 + var.LSTM_weight_2(:,:,1,6).' * diff_i_2 + var.LSTM_weight_2(:,:,1,7).' * diff_f_2 + var.LSTM_weight_2(:,:,1,8).' * diff_o_2;
 
        diff_forward_output_1_t = gradient(:,:,sys.tx+1-i) + diff_forward_output_1_h;
        if i == 1
            diff_forward_output_1_state = diff_forward_output_1_t .* var.gate_output(:,:,sys.tx+1-i,4,1,1) .* (1 - tanh(var.hidden_state(:,:,sys.tx+2-i,1,1)).^2); 
        else
            diff_forward_output_1_state = diff_forward_output_1_t .* var.gate_output(:,:,sys.tx+1-i,4,1,1) .* (1 - tanh(var.hidden_state(:,:,sys.tx+2-i,1,1)).^2) + diff_forward_output_1_state .* var.gate_output(:,:,sys.tx+2-i,3,1,1);          
        end
        
        diff_a_1 = diff_forward_output_1_state .* var.gate_output(:,:,sys.tx+1-i,2,1,1) .* (1 - var.gate_output(:,:,sys.tx+1-i,1,1,1).^2);
        diff_i_1 = diff_forward_output_1_state .* var.gate_output(:,:,sys.tx+1-i,1,1,1) .* var.gate_output(:,:,sys.tx+1-i,2,1,1) .* (1 - var.gate_output(:,:,sys.tx+1-i,2,1,1));
        diff_f_1 = diff_forward_output_1_state .* var.hidden_state(:,:,sys.tx+1-i,1,1) .* var.gate_output(:,:,sys.tx+1-i,3,1,1) .* (1 - var.gate_output(:,:,sys.tx+1-i,3,1,1));
        diff_o_1 = diff_forward_output_1_t .* tanh(var.hidden_state(:,:,sys.tx+2-i,1,1)) .* var.gate_output(:,:,sys.tx+1-i,4,1,1) .* (1 - var.gate_output(:,:,sys.tx+1-i,4,1,1));
    
        var.LSTM_weight_1(:,:,2,1) = var.LSTM_weight_1(:,:,2,1) + diff_a_1 * input(:,:,sys.tx+1-i).';
        var.LSTM_weight_1(:,:,2,2) = var.LSTM_weight_1(:,:,2,2) + diff_i_1 * input(:,:,sys.tx+1-i).';
        var.LSTM_weight_1(:,:,2,3) = var.LSTM_weight_1(:,:,2,3) + diff_f_1 * input(:,:,sys.tx+1-i).';
        var.LSTM_weight_1(:,:,2,4) = var.LSTM_weight_1(:,:,2,4) + diff_o_1 * input(:,:,sys.tx+1-i).';
        
        var.LSTM_weight_2(:,:,2,1) = var.LSTM_weight_2(:,:,2,1) + diff_a_1 * var.hidden_output(:,:,sys.tx+1-i,1,1).';
        var.LSTM_weight_2(:,:,2,2) = var.LSTM_weight_2(:,:,2,2) + diff_i_1 * var.hidden_output(:,:,sys.tx+1-i,1,1).';
        var.LSTM_weight_2(:,:,2,3) = var.LSTM_weight_2(:,:,2,3) + diff_f_1 * var.hidden_output(:,:,sys.tx+1-i,1,1).';
        var.LSTM_weight_2(:,:,2,4) = var.LSTM_weight_2(:,:,2,4) + diff_o_1 * var.hidden_output(:,:,sys.tx+1-i,1,1).';
        
        var.LSTM_bias_b(:,2,1) = var.LSTM_bias_b(:,2,1) + sum(diff_a_1,2);
        var.LSTM_bias_b(:,2,2) = var.LSTM_bias_b(:,2,2) + sum(diff_i_1,2);
        var.LSTM_bias_b(:,2,3) = var.LSTM_bias_b(:,2,3) + sum(diff_f_1,2);
        var.LSTM_bias_b(:,2,4) = var.LSTM_bias_b(:,2,4) + sum(diff_o_1,2);
        
        diff_output_2(:,:,sys.tx+1-i) = var.LSTM_weight_1(:,:,1,1).' * diff_a_1 + var.LSTM_weight_1(:,:,1,2).' * diff_i_1 + var.LSTM_weight_1(:,:,1,3).' * diff_f_1 + var.LSTM_weight_1(:,:,1,4).' * diff_o_1;
        diff_forward_output_1_h = var.LSTM_weight_2(:,:,1,1).' * diff_a_1 + var.LSTM_weight_2(:,:,1,2).' * diff_i_1 + var.LSTM_weight_2(:,:,1,3).' * diff_f_1 + var.LSTM_weight_2(:,:,1,4).' * diff_o_1;
    end    
    gradient_hat = (diff_output_1 + diff_output_2)./2; 
    
 end

 clear diff_output_1 diff_output_2
end