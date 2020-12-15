function [a_output_2,gradient] = dense_output_layer(input,samples_com)

global sys;
global var;

    for i = 1:sys.tx
        z_output_1 = var.LSTM_weight_2(:,:,1,end) * input(:,:,i) + kron(var.LSTM_bias_b(:,1,end),ones(1,sys.batch_size));
        a_output_1 = max(0,z_output_1);
        jac_1_k = a_output_1 > 0;
        
        z_output_2 = var.dense_weight(:,:,1) * a_output_1 + kron(var.dense_bias(:,1),ones(1,sys.batch_size));
        a_output_2(:,:,i) = 1 ./ (1+exp(-z_output_2));
 
        diff_final_output = a_output_2(:,:,i) - samples_com(:,:,i);   

        var.dense_weight(:,:,2) = var.dense_weight(:,:,2) + diff_final_output * a_output_1.';  
        var.dense_bias(:,2) = var.dense_bias(:,2) + sum(diff_final_output,2);   
        
        diff_z_output_1 = var.dense_weight(:,:,1).' * diff_final_output .* jac_1_k;
        
        var.LSTM_weight_2(:,:,2,end) = var.LSTM_weight_2(:,:,2,end) + diff_z_output_1 * input(:,:,i).';   
        var.LSTM_bias_b(:,2,end) = var.LSTM_bias_b(:,2,end) + sum(diff_z_output_1,2);      

        gradient(:,:,i) = var.LSTM_weight_2(:,:,1,end).' * diff_z_output_1; 
         
    end

end