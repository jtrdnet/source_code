function gradient_hat = backpropagation_dense(final_output,samples_com,received_signal)

global sys;
global var;

    var.samples_com_hat = [];
    for i = 1:sys.tx
        var.samples_com_hat = [var.samples_com_hat;samples_com(:,:,i)];
    end
    
    gradient_4 = final_output - var.samples_com_hat;
    var.output_layer(:,:,2) = gradient_4 * var.a_3.' ./ sys.batch_size;
    var.output_layer_bias(:,2) = sum(gradient_4,2)./ sys.batch_size;
 
    gradient_3 = var.output_layer(:,:,1).' * gradient_4 .* var.jac_3;
    var.hidden_layer_2(:,:,2) = gradient_3 * var.a_2.' ./ sys.batch_size;
    var.hidden_layer_2_bias(:,2) = sum(gradient_3,2)./ sys.batch_size;
    
    gradient_2 = var.hidden_layer_2(:,:,1).' * gradient_3 .* var.jac_2;
    var.hidden_layer_1(:,:,2) = gradient_2 * var.a_1.' ./ sys.batch_size;
    var.hidden_layer_1_bias(:,2) = sum(gradient_2,2)./ sys.batch_size;
    
    gradient_1 = var.hidden_layer_1(:,:,1).' * gradient_2 .* var.jac_1;
    var.input_layer(:,:,2) = gradient_1 * received_signal.' ./ sys.batch_size;
    var.input_layer_bias(:,2) = sum(gradient_1,2)./ sys.batch_size;
    
    gradient_hat = var.input_layer(:,:,1).' * gradient_1;
end