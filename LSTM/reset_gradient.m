function reset_gradient()

global var; 
global sys;

if sys.GPU == 0 
    var.LSTM_weight_1(:,:,2,:) = zeros(size(var.LSTM_weight_1(:,:,2,:)));
    var.LSTM_weight_2(:,:,2,:) = zeros(size(var.LSTM_weight_2(:,:,2,:)));
    var.LSTM_bias_b(:,2,:) = zeros(size(var.LSTM_bias_b(:,2,:)));
    var.dense_weight(:,:,2) = zeros(size(var.dense_weight(:,:,3)));
    var.dense_bias(:,2) = zeros(size(var.dense_bias(:,3)));
    var.linear_layer(:,:,2,:) = zeros(size(var.linear_layer(:,:,2,:)));
elseif sys.GPU == 1
    warning('GPU function undefined in this version!')
    
end

end