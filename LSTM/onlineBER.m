function onlineBER(final_output,samples_com,iteration)

global sys;
 
for i = 1:sys.tx
    sys.counter = sys.counter + sum(sum(round(final_output(:,:,i)) ~= samples_com(:,:,i)));
end
    if mod(iteration,sys.stack_size+1) ~= 0
        sys.Err(mod(iteration,sys.stack_size+1)) = sys.counter;                                    
    end
    
    BER = sum(sys.Err,2)./(sys.stack_size * sys.batch_size * sys.data_rate * sys.tx)  
   

end