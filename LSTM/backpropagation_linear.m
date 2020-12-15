function backpropagation_linear(gradient_LSTM_1,information_bearing_bits)

global sys;
global var;

    for i = 1:sys.time_dof
        diff_linear(:,:,i) = var.H.' * gradient_LSTM_1(:,:,i);
    end

    for i = 1:sys.tx
        a = [];
        for j = 1:sys.time_dof
            a = [a;diff_linear(i,:,j)];
            a = [a;diff_linear(i+sys.tx,:,j)];
        end   
        var.linear_layer(:,:,2,i) = ( a .* var.jac_1(:,:,i)) * information_bearing_bits(:,:,i).'./ sys.batch_size; 
    end
end