function parameter_initilization()
global sys;
global var;
    if sys.GPU == 1
            warning('GPU function undefined in the current version!')
    elseif sys.GPU == 0 
            var.linear_layer = (rand(2*sys.time_dof,2^sys.data_rate,1,sys.tx)-0.5)*0.02;
            var.input_layer = (rand(sys.hidden_dimension,sys.input_dimension)-0.5)*0.02;
            var.input_layer_bias = zeros(sys.hidden_dimension,6);
            var.hidden_layer_1 = (rand(sys.hidden_dimension/2,sys.hidden_dimension)-0.5)*0.2; 
            var.hidden_layer_1_bias = zeros(sys.hidden_dimension./2,6);
            var.hidden_layer_2 = (rand(sys.hidden_dimension/4,sys.hidden_dimension/2)-0.5)*0.2; 
            var.hidden_layer_2_bias = zeros(sys.hidden_dimension./4,6);
            var.output_layer = (rand(sys.output_dimension,sys.hidden_dimension./4)-0.5)*0.2;
            var.output_layer_bias = zeros(sys.output_dimension,6);
            for i = 2:6
               var.linear_layer(:,:,i,:) = zeros(2*sys.time_dof,2^sys.data_rate,1,sys.tx);
               var.input_layer(:,:,i) = zeros(sys.hidden_dimension,sys.input_dimension);
               var.hidden_layer_1(:,:,i) = zeros(sys.hidden_dimension/2,sys.hidden_dimension);
               var.hidden_layer_2(:,:,i) = zeros(sys.hidden_dimension/4,sys.hidden_dimension/2);
               var.output_layer = zeros(sys.output_dimension,sys.hidden_dimension/4,i);
            end
            
    end
end