"""
Global Setting of project IS712
"""

TOY = False  # Test code with toy data
SEPARATE = False  # Separate parameters of mean and variance prediction
MEAN_MODEL_NAME = "resnet"
VAR_MODEL_NAME = ""
UPDATE_RESNET = True
DISCRETE_REG = False  # Use regression and map result into discrete_means/discrete_vars
DISCRETE_CLS = False  # Use classification with classes in discrete_means/discrete_vars
IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 16 if not TOY else 2
EPOCH_NUM = 100 if not TOY else 10
GPU_NUM = 2
VAL_FOLD_ID = 0
ROUND_NUM = 5
CONVERGE_INCREMENT_THRESHOLD = 1e-5  # Increment threshold for early stopping
CONVERGE_EPOCH_NUM_THRESHOLD = 10  # Epoch number threshold for early stopping
FINE_TUNE_LR = 1e-4
NEW_PARAM_LR = 1e-3
WEIGHT_DECAY = 5e-4
BEST_MODEL_PATH = "./model/best_model_fold_%d" % VAL_FOLD_ID
RES_PATH = "submission_%d.txt" % VAL_FOLD_ID
device = 'cuda'
TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/validation'

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
print(head % "CONFIGURATION")
print(head % "TOY", TOY)
print(head % "MEAN_MODEL_NAME", MEAN_MODEL_NAME)
print(head % "VAR_MODEL_NAME", VAR_MODEL_NAME)
print(head % "SEPARATE", SEPARATE)
print(head % "UPDATE_RESNET", UPDATE_RESNET)
print(head % "CONVERGE_INCREMENT_THRESHOLD", CONVERGE_INCREMENT_THRESHOLD)
print(head % "CONVERGE_EPOCH_NUM_THRESHOLD", CONVERGE_EPOCH_NUM_THRESHOLD)
print(head % "FINE_TUNE_LR", FINE_TUNE_LR)
print(head % "NEW_PARAM_LR", NEW_PARAM_LR)
print(head % "WEIGHT_DECAY", WEIGHT_DECAY)
print(head % "DISCRETE_REG", DISCRETE_REG)
print(head % "DISCRETE_CLS", DISCRETE_CLS)
print(head % "BATCH_SIZE", BATCH_SIZE)
print(head % "EPOCH_NUM", EPOCH_NUM)
print(head % "VAL_FOLD_ID", VAL_FOLD_ID)
print(head % "GPU_NUM", GPU_NUM)
print(head % "BEST_MODEL_PATH", BEST_MODEL_PATH)
print(head % "RES_PATH", RES_PATH)
print("")
