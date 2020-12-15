function [H,noise] = generate_channel()

global sys;
global var;

    if sys.GPU == 0
        if strncmpi(sys.channel_type,'Rayleigh',8)==1
            var.H = 1./sqrt(2*sys.tx) * (randn(sys.rx,sys.tx) + 1i * randn(sys.rx,sys.tx)); 
            var.noise = 1./sqrt(2) * (randn(sys.rx,sys.batch_size,sys.time_dof) + 1i * randn(sys.rx,sys.batch_size,sys.time_dof))./sys.noise_var_training;
            var.H = [real(var.H) -imag(var.H);imag(var.H) real(var.H)];
        else
            warning('channle type undefined!');
        end
    else
        if strncmpi(sys.channel_type,'Rayleigh',8)==1
            var.H = gpuArray(1./sqrt(2*sys.tx) * (randn(sys.rx,sys.tx) + 1i * randn(sys.rx,sys.tx))); 
            var.noise = gpuArray(1./sqrt(2) * (randn(sys.rx,sys.batch_size) + 1i * randn(sys.rx,sys.batch_size))./sys.noise_var_training);
            var.H = [real(var.H) -imag(var.H);imag(var.H) real(var.H)];
        else
            warning('channle type undefined!');
        end
    end  
end