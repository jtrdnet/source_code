function onlineBER(final_output,iteration)

global sys;
global var;

    sys.counter = sum(sum(round(final_output) ~= var.samples_com_hat));

    if mod(iteration,sys.stack_size+1) ~= 0
        sys.Err(mod(iteration,sys.stack_size+1)) = sys.counter;                                    
    end
    
    BER = sum(sys.Err,2)./(sys.stack_size * sys.batch_size * sys.data_rate * sys.tx)  
   

end