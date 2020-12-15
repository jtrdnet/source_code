function adam(iteration)

global sys;
global var;

var.input_layer(:,:,3) = sys.beta1 * var.input_layer(:,:,3) + (1 - sys.beta1) * var.input_layer(:,:,2);
var.input_layer(:,:,4) = var.input_layer(:,:,3) ./ (1 - sys.beta1.^iteration);
var.input_layer(:,:,5) = sys.beta2 * var.input_layer(:,:,5) + (1 - sys.beta2) * var.input_layer(:,:,2).^2;
var.input_layer(:,:,6) = var.input_layer(:,:,5) ./ (1 - sys.beta2.^iteration);
var.input_layer(:,:,1) = var.input_layer(:,:,1) - sys.lr * var.input_layer(:,:,4) ./ (sqrt(var.input_layer(:,:,6)) + sys.eps);

var.input_layer_bias(:,3) = sys.beta1 * var.input_layer_bias(:,3) + (1 - sys.beta1) * var.input_layer_bias(:,2);
var.input_layer_bias(:,4) = var.input_layer_bias(:,3) ./ (1 - sys.beta1.^iteration);
var.input_layer_bias(:,5) = sys.beta2 * var.input_layer_bias(:,5) + (1 - sys.beta2) * var.input_layer_bias(:,2).^2;
var.input_layer_bias(:,6) = var.input_layer_bias(:,5) ./ (1 - sys.beta2.^iteration);
var.input_layer_bias(:,1) = var.input_layer_bias(:,1) - sys.lr * var.input_layer_bias(:,4) ./ (sqrt(var.input_layer_bias(:,6)) + sys.eps);

var.hidden_layer_1(:,:,3) = sys.beta1 * var.hidden_layer_1(:,:,3) + (1 - sys.beta1) * var.hidden_layer_1(:,:,2);
var.hidden_layer_1(:,:,4) = var.hidden_layer_1(:,:,3) ./ (1 - sys.beta1.^iteration);
var.hidden_layer_1(:,:,5) = sys.beta2 * var.hidden_layer_1(:,:,5) + (1 - sys.beta2) * var.hidden_layer_1(:,:,2).^2;
var.hidden_layer_1(:,:,6) = var.hidden_layer_1(:,:,5) ./ (1 - sys.beta2.^iteration);
var.hidden_layer_1(:,:,1) = var.hidden_layer_1(:,:,1) - sys.lr * var.hidden_layer_1(:,:,4) ./ (sqrt(var.hidden_layer_1(:,:,6)) + sys.eps);

var.hidden_layer_1_bias(:,3) = sys.beta1 * var.hidden_layer_1_bias(:,3) + (1 - sys.beta1) * var.hidden_layer_1_bias(:,2);
var.hidden_layer_1_bias(:,4) = var.hidden_layer_1_bias(:,3) ./ (1 - sys.beta1.^iteration);
var.hidden_layer_1_bias(:,5) = sys.beta2 * var.hidden_layer_1_bias(:,5) + (1 - sys.beta2) * var.hidden_layer_1_bias(:,2).^2;
var.hidden_layer_1_bias(:,6) = var.hidden_layer_1_bias(:,5) ./ (1 - sys.beta2.^iteration);
var.hidden_layer_1_bias(:,1) = var.hidden_layer_1_bias(:,1) - sys.lr * var.hidden_layer_1_bias(:,4) ./ (sqrt(var.hidden_layer_1_bias(:,6)) + sys.eps);

var.hidden_layer_2(:,:,3) = sys.beta1 * var.hidden_layer_2(:,:,3) + (1 - sys.beta1) * var.hidden_layer_2(:,:,2);
var.hidden_layer_2(:,:,4) = var.hidden_layer_2(:,:,3) ./ (1 - sys.beta1.^iteration);
var.hidden_layer_2(:,:,5) = sys.beta2 * var.hidden_layer_2(:,:,5) + (1 - sys.beta2) * var.hidden_layer_2(:,:,2).^2;
var.hidden_layer_2(:,:,6) = var.hidden_layer_2(:,:,5) ./ (1 - sys.beta2.^iteration);
var.hidden_layer_2(:,:,1) = var.hidden_layer_2(:,:,1) - sys.lr * var.hidden_layer_2(:,:,4) ./ (sqrt(var.hidden_layer_2(:,:,6)) + sys.eps);

var.hidden_layer_2_bias(:,3) = sys.beta1 * var.hidden_layer_2_bias(:,3) + (1 - sys.beta1) * var.hidden_layer_2_bias(:,2);
var.hidden_layer_2_bias(:,4) = var.hidden_layer_2_bias(:,3) ./ (1 - sys.beta1.^iteration);
var.hidden_layer_2_bias(:,5) = sys.beta2 * var.hidden_layer_2_bias(:,5) + (1 - sys.beta2) * var.hidden_layer_2_bias(:,2).^2;
var.hidden_layer_2_bias(:,6) = var.hidden_layer_2_bias(:,5) ./ (1 - sys.beta2.^iteration);
var.hidden_layer_2_bias(:,1) = var.hidden_layer_2_bias(:,1) - sys.lr * var.hidden_layer_2_bias(:,4) ./ (sqrt(var.hidden_layer_2_bias(:,6)) + sys.eps);

var.output_layer(:,:,3) = sys.beta1 * var.output_layer(:,:,3) + (1 - sys.beta1) * var.output_layer(:,:,2);
var.output_layer(:,:,4) = var.output_layer(:,:,3) ./ (1 - sys.beta1.^iteration);
var.output_layer(:,:,5) = sys.beta2 * var.output_layer(:,:,5) + (1 - sys.beta2) * var.output_layer(:,:,2).^2;
var.output_layer(:,:,6) = var.output_layer(:,:,5) ./ (1 - sys.beta2.^iteration);
var.output_layer(:,:,1) = var.output_layer(:,:,1) - sys.lr * var.output_layer(:,:,4) ./ (sqrt(var.output_layer(:,:,6)) + sys.eps);

var.output_layer_bias(:,3) = sys.beta1 * var.output_layer_bias(:,3) + (1 - sys.beta1) * var.output_layer_bias(:,2);
var.output_layer_bias(:,4) = var.output_layer_bias(:,3) ./ (1 - sys.beta1.^iteration);
var.output_layer_bias(:,5) = sys.beta2 * var.output_layer_bias(:,5) + (1 - sys.beta2) * var.output_layer_bias(:,2).^2;
var.output_layer_bias(:,6) = var.output_layer_bias(:,5) ./ (1 - sys.beta2.^iteration);
var.output_layer_bias(:,1) = var.output_layer_bias(:,1) - sys.lr * var.output_layer_bias(:,4) ./ (sqrt(var.output_layer_bias(:,6)) + sys.eps);

var.linear_layer(:,:,3,:) = sys.beta1 * var.linear_layer(:,:,3,:) + (1 - sys.beta1) * var.linear_layer(:,:,2,:);
var.linear_layer(:,:,4,:) = var.linear_layer(:,:,3,:) ./ (1 - sys.beta1.^iteration);
var.linear_layer(:,:,5,:) = sys.beta2 * var.linear_layer(:,:,5,:) + (1 - sys.beta2) * var.linear_layer(:,:,2,:).^2;
var.linear_layer(:,:,6,:) = var.linear_layer(:,:,5,:) ./ (1 - sys.beta2.^iteration);
var.linear_layer(:,:,1,:) = var.linear_layer(:,:,1,:) - sys.lr * var.linear_layer(:,:,4,:) ./ (sqrt(var.linear_layer(:,:,6,:)) + sys.eps);
   
end