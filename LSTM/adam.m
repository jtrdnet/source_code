function adam(iteration)

global sys;
global var;

var.LSTM_weight_1(:,:,3,:) = sys.beta1 * var.LSTM_weight_1(:,:,3,:) + (1 - sys.beta1) * var.LSTM_weight_1(:,:,2,:);
var.LSTM_weight_1(:,:,4,:) = var.LSTM_weight_1(:,:,3,:) ./ (1 - sys.beta1.^iteration);
var.LSTM_weight_1(:,:,5,:) = sys.beta2 * var.LSTM_weight_1(:,:,5,:) + (1 - sys.beta2) * var.LSTM_weight_1(:,:,2,:).^2;
var.LSTM_weight_1(:,:,6,:) = var.LSTM_weight_1(:,:,5,:) ./ (1 - sys.beta2.^iteration);
var.LSTM_weight_1(:,:,1,:) = var.LSTM_weight_1(:,:,1,:) - sys.lr * var.LSTM_weight_1(:,:,4,:) ./ (sqrt(var.LSTM_weight_1(:,:,6,:)) + sys.eps);
   
var.LSTM_weight_2(:,:,3,:) = sys.beta1 * var.LSTM_weight_2(:,:,3,:) + (1 - sys.beta1) * var.LSTM_weight_2(:,:,2,:);
var.LSTM_weight_2(:,:,4,:) = var.LSTM_weight_2(:,:,3,:) ./ (1 - sys.beta1.^iteration);
var.LSTM_weight_2(:,:,5,:) = sys.beta2 * var.LSTM_weight_2(:,:,5,:) + (1 - sys.beta2) * var.LSTM_weight_2(:,:,2,:).^2;
var.LSTM_weight_2(:,:,6,:) = var.LSTM_weight_2(:,:,5,:) ./ (1 - sys.beta2.^iteration);
var.LSTM_weight_2(:,:,1,:) = var.LSTM_weight_2(:,:,1,:) - sys.lr * var.LSTM_weight_2(:,:,4,:) ./ (sqrt(var.LSTM_weight_2(:,:,6,:)) + sys.eps);

var.LSTM_bias_b(:,3,:) = sys.beta1 * var.LSTM_bias_b(:,3,:) + (1 - sys.beta1) * var.LSTM_bias_b(:,2,:);
var.LSTM_bias_b(:,4,:) = var.LSTM_bias_b(:,3,:) ./ (1 - sys.beta1.^iteration);
var.LSTM_bias_b(:,5,:) = sys.beta2 * var.LSTM_bias_b(:,5,:) + (1 - sys.beta2) * var.LSTM_bias_b(:,2,:).^2;
var.LSTM_bias_b(:,6,:) = var.LSTM_bias_b(:,5,:) ./ (1 - sys.beta2.^iteration);
var.LSTM_bias_b(:,1,:) = var.LSTM_bias_b(:,1,:) - sys.lr * var.LSTM_bias_b(:,4,:) ./ (sqrt(var.LSTM_bias_b(:,6,:)) + sys.eps);

var.dense_weight(:,:,3) = sys.beta1 * var.dense_weight(:,:,3) + (1 - sys.beta1) * var.dense_weight(:,:,2);
var.dense_weight(:,:,4) = var.dense_weight(:,:,3) ./ (1 - sys.beta1.^iteration);
var.dense_weight(:,:,5) = sys.beta2 * var.dense_weight(:,:,5) + (1 - sys.beta2) * var.dense_weight(:,:,2).^2;
var.dense_weight(:,:,6) = var.dense_weight(:,:,5) ./ (1 - sys.beta2.^iteration);
var.dense_weight(:,:,1) = var.dense_weight(:,:,1) - sys.lr * var.dense_weight(:,:,4) ./ (sqrt(var.dense_weight(:,:,6)) + sys.eps);

var.dense_bias(:,3) = sys.beta1 * var.dense_bias(:,3) + (1 - sys.beta1) * var.dense_bias(:,2);
var.dense_bias(:,4) = var.dense_bias(:,3) ./ (1 - sys.beta1.^iteration);
var.dense_bias(:,5) = sys.beta2 * var.dense_bias(:,5) + (1 - sys.beta2) * var.dense_bias(:,2).^2;
var.dense_bias(:,6) = var.dense_bias(:,5) ./ (1 - sys.beta2.^iteration);
var.dense_bias(:,1) = var.dense_bias(:,1) - sys.lr * var.dense_bias(:,4) ./ (sqrt(var.dense_bias(:,6)) + sys.eps);

var.linear_layer(:,:,3,:) = sys.beta1 * var.linear_layer(:,:,3,:) + (1 - sys.beta1) * var.linear_layer(:,:,2,:);
var.linear_layer(:,:,4,:) = var.linear_layer(:,:,3,:) ./ (1 - sys.beta1.^iteration);
var.linear_layer(:,:,5,:) = sys.beta2 * var.linear_layer(:,:,5,:) + (1 - sys.beta2) * var.linear_layer(:,:,2,:).^2;
var.linear_layer(:,:,6,:) = var.linear_layer(:,:,5,:) ./ (1 - sys.beta2.^iteration);
var.linear_layer(:,:,1,:) = var.linear_layer(:,:,1,:) - sys.lr * var.linear_layer(:,:,4,:) ./ (sqrt(var.linear_layer(:,:,6,:)) + sys.eps);
   
end