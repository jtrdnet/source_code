function transmitted_signal = linear_layer(information_bearing_bits)

global sys;
global var;
     
    for i = 1:sys.tx 
        z_layer1(:,:,i) = var.linear_layer(:,:,1,i) * information_bearing_bits(:,:,i);
        norm_vec(:,:,i) = sqrt(sum(z_layer1(:,:,i).^2));
        beta_matrix(:,:,i) = kron(ones(2*sys.time_dof,1),norm_vec(:,:,i));
        var.a_norm(:,:,i) = z_layer1(:,:,i) * sqrt(sys.time_dof) ./ beta_matrix(:,:,i);
        var.jac_l(:,:,i) = sqrt(sys.time_dof) * (beta_matrix(:,:,i).^2 - z_layer1(:,:,i).^2)./(beta_matrix(:,:,i).^3);
    end
    
    transmitted_signal = zeros(2*sys.tx,sys.batch_size,sys.time_dof);
    for i = 1:sys.time_dof
        slot = [];
        for j = 1:sys.tx
            slot = [slot;var.a_norm((2*i-1),:,j)];
        end
        for j = 1:sys.tx
            slot = [slot;var.a_norm((2*i),:,j)];
        end
        transmitted_signal(:,:,i) = slot;
    end



end