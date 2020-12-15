function reset_gradient()

global var; 
global sys;

if sys.GPU == 0 
    var.input_layer(:,:,2) = zeros(size(var.input_layer(:,:,2)));
    var.input_layer_bias(:,2) = zeros(size(var.input_layer(:,2)));
    var.hidden_layer_1(:,:,2) = zeros(size(var.hidden_layer_1(:,:,2)));
    var.hidden_layer_1_bias(:,2) = zeros(size(var.hidden_layer_1_bias(:,2)));
    var.hidden_layer_2(:,:,2) = zeros(size(var.hidden_layer_2(:,:,2)));
    var.hidden_layer_2_bias(:,2) = zeros(size(var.hidden_layer_2_bias(:,2)));
    var.output_layer(:,:,2) = zeros(size(var.output_layer(:,:,2)));
    var.output_layer_bias(:,2) = zeros(size(var.output_layer(:,2)));
    var.linear_layer(:,:,2,:) = zeros(size(var.linear_layer(:,:,2,:)));
elseif sys.GPU == 1
    warning('GPU function undefined in this version!')
    
end

end