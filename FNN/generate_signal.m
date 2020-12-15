function [information_bearing_bits,samples_com] = generate_signal()

global sys;
global var;
    if sys.GPU == 0
        for i = 1:sys.batch_size
            for j = 1:sys.tx
                index(j,i) = randperm(2^sys.data_rate,1); 
                samples_com(:,i,j) = var.sample(:,index(j,i)); 
                information_bearing_bits(:,i,j) = var.id(:,index(j,i));
            end
        end
    elseif sys.GPU == 1
        for i = 1:sys.batch_size
            for j = 1:sys.tx
                index(j,i) = randperm(2^sys.mod_level,1); 
                samples_com(:,i,j) = gpuArray(var.sample(:,index(j,i))); 
                information_bearing_bits(:,i,j) = gpuArray(var.id(:,index(j,i)));
            end
        end
    else 
        warning('GPU type undefined!');
    end   
end