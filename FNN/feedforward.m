function a_4 = feedforward(input)

global sys;
global var;
     
    z_1 = var.input_layer(:,:,1) * input + kron(var.input_layer_bias(:,1),ones(1,sys.batch_size));
    var.a_1 = max(z_1,0);
    var.jac_1 = var.a_1 > 0;
    
    z_2 = var.hidden_layer_1(:,:,1) * var.a_1 + kron(var.hidden_layer_1_bias(:,1),ones(1,sys.batch_size));
    var.a_2 = max(z_2,0);
    var.jac_2 = var.a_2 > 0;
    
    z_3 = var.hidden_layer_2(:,:,1) * var.a_2 + kron(var.hidden_layer_2_bias(:,1),ones(1,sys.batch_size));
    var.a_3 = max(z_3,0);
    var.jac_3 = var.a_3 > 0; 
    
    z_4 = var.output_layer(:,:,1) * var.a_3 + kron(var.output_layer_bias(:,1),ones(1,sys.batch_size));
    a_4 = 1./(1+exp(-z_4));
end
