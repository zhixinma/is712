"""
Global Setting of project IS712
"""

toy = False  # Test code with toy data
separate = True  # Separate parameters of mean and variance prediction
mean_model_name = "cnn"
var_model_name = "cnn"
update_resnet = True
use_fea = False  # Use handcraft feature (e.g. RGB Histogram)
discrete_reg = False  # Use regression and map result into discrete_means/discrete_vars
discrete_cls = False  # Use classification with classes in discrete_means/discrete_vars
IMG_SHAPE = (256, 256, 3)
batch_size = 128 if not toy else 2
num_epoch = 200 if not toy else 10
gpu_num = 3
val_fold_id = 0
round_num = 5
CONVERGE_INCREMENT_THRESHOLD = 1e-5  # Increment threshold for early stopping
CONVERGE_EPOCH_NUM_THRESHOLD = 10  # Epoch number threshold for early stopping
best_model_path = "./model/best_model_fold_%d" % val_fold_id
res_path = "submission_%d.txt" % val_fold_id
device = 'cuda'
train_dir = './dataset/train'
test_dir = './dataset/validation'

discrete_means = [1.67, 2.33, 2.67, 3.33, 3.67, 4.33, 4.67, 5.0, 5.33, 5.67, 6.0, 6.33, 6.67, 7.0, 7.33, 7.67, 8.0,
                  8.33, 8.66, 8.67, 9.33]
discrete_vars = [0.471416306, 0.471451659, 0.816496581, 0.942814934, 1.247223583, 1.414213562, 1.632993162, 1.69967644,
                 1.885621029, 2.054807371]
discrete_vars = [round(i, 4) for i in discrete_vars]
discrete_mean_map = {discrete_means[i]: i for i in range(len(discrete_means))}
discrete_var_map = {discrete_vars[i]: i for i in range(len(discrete_vars))}
MEAN_CLS_NUM = len(discrete_means)
VAR_CLS_NUM = len(discrete_vars)

head = "%30s | "
print(head % "Configuration")
print(head % "toy", toy)
print(head % "mean_model_name", mean_model_name)
print(head % "var_model_name", var_model_name)
print(head % "separate", separate)
print(head % "update_resnet", update_resnet)
print(head % "CONVERGE_INCREMENT_THRESHOLD", CONVERGE_INCREMENT_THRESHOLD)
print(head % "CONVERGE_EPOCH_NUM_THRESHOLD", CONVERGE_EPOCH_NUM_THRESHOLD)
print(head % "use_hist", use_fea)
print(head % "discrete_reg", discrete_reg)
print(head % "discrete_cls", discrete_cls)
print(head % "batch_size", batch_size)
print(head % "num_epoch", num_epoch)
print(head % "val_fold_id", val_fold_id)
print(head % "gpu_num", gpu_num)
print(head % "best_model_path", best_model_path)
print(head % "res_path", res_path)
print("")
