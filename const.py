"""
Global Setting of project IS712
"""

TOY = False  # Test code with toy data
SEPARATE = False  # Separate parameters of mean and variance prediction
USE_FULL_DATA = False
USE_KNN_VALID = False
MEAN_MODEL_NAME = "resnet"
VAR_MODEL_NAME = ""
UPDATE_RESNET = True
DISCRETE_REG = False  # Use regression and map result into discrete_means/discrete_vars
DISCRETE_CLS = False  # Use classification with classes in discrete_means/discrete_vars
GPU_NUM = 0
VAL_FOLD_ID = 0
PREFIX = "resnet_lrs_val"
IMG_SHAPE = (256, 256, 3)
BATCH_SIZE = 16 if not TOY else 2
FINE_TUNE_EPOCH = 0
EPOCH_NUM = (30 if USE_FULL_DATA else 100) if not TOY else 10
ROUND_NUM = 5
CONVERGE_INCREMENT_THRESHOLD = 1e-5  # Increment threshold for early stopping
CONVERGE_EPOCH_NUM_THRESHOLD = 10  # Epoch number threshold for early stopping
FINE_TUNE_LR = 1e-4
NEW_PARAM_LR = 1e-3
WEIGHT_DECAY = 5e-4
BEST_MODEL_PATH = "./model/best_%s_%d" % (PREFIX, VAL_FOLD_ID)
RES_PATH = "submission_test_%s_%d.txt" % (PREFIX, VAL_FOLD_ID)
DEVICE = 'cuda'
TRAIN_DIR = './dataset/train'
VALID_DIR = './dataset/validation'
TEST_DIR = './dataset/test'

discrete_means = [1.67, 2.33, 2.67, 3.33, 3.67, 4.33, 4.67, 5.0, 5.33, 5.67, 6.0, 6.33, 6.67, 7.0, 7.33, 7.67, 8.0,
                  8.33, 8.66, 8.67, 9.33]
discrete_vars = [0.471416306, 0.471451659, 0.816496581, 0.942814934, 1.247223583, 1.414213562, 1.632993162, 1.69967644,
                 1.885621029, 2.054807371]
VALID_IDS_KNN = [432, 1521, 2400, 2770, 90, 513, 50, 2930, 1219, 1609, 1437, 2219, 223, 2525, 1053, 2685, 56, 668, 540, 2644, 2744, 781, 731, 774, 1377, 2371, 263, 1595, 101, 803, 207, 1438, 646, 219, 1741, 939, 2301, 710, 2776, 794, 2861, 1453, 2430, 1566, 2651, 1269, 559, 955, 1347, 663, 954, 2079, 416, 211, 2723, 2355, 1485, 1524, 2692, 25, 71, 149, 2073, 1537, 2700, 2023, 966, 2473, 464, 1335, 2813, 308, 1163, 2534, 24, 1420, 1413, 2795, 2886, 2001, 39, 519, 1281, 1782, 512, 179, 713, 2169, 202, 1761, 763, 190, 157, 2565, 1492, 1165, 2868, 1488, 241, 396, 911, 287, 255, 799, 2372, 991, 435, 462, 1831, 805, 2955, 2420, 807, 376, 1329, 2359, 348, 363, 358, 1771, 2802, 2320, 1012, 1489, 117, 969, 1399, 2820, 2191, 440, 1902, 2283, 85, 1396, 2960, 1811, 1517, 1865, 483, 1427, 973, 2183, 1715, 2307, 893, 329, 549, 1345, 1400, 2936, 1470, 1695, 915, 233, 2140, 2986, 2441, 1188, 995, 1900, 2225, 912, 2475, 878, 259, 447, 595, 2334, 2446, 2299, 380, 565, 2915, 271, 682, 2559, 2443, 847, 2208, 1062, 1916, 1035, 1854, 402, 1591, 2135, 2349, 2165, 2789, 195, 864, 497, 524, 2303, 2687, 962, 2102, 1739, 2332, 1988, 188, 2821, 489, 1910, 1588, 860, 804, 2077, 770, 2831, 2873, 1128, 938, 385, 1454, 2078, 2638, 634, 2029, 2000, 1848, 1069, 2569, 177, 2693, 1258, 1306, 1498, 1838, 1363, 706, 1677, 217, 2043, 2547, 270, 886, 1037, 1403, 1888, 1474, 2, 1007, 2316, 2561, 2998, 2428, 1066, 305, 788, 1861, 64, 1712, 126, 1620, 2438, 681, 198, 369, 162, 2898, 580, 451, 2794, 423, 1814, 2374, 257, 1300, 2510, 1047, 499, 1520, 2389, 2351, 1460, 854, 2507, 572, 359, 1379, 1603, 783, 1769, 88, 696, 2010, 1013, 1534, 2798, 2007, 1175, 2680, 1696, 2921, 1231, 399, 1014, 1646, 1570]
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
print(head % "USE_FULL_DATA", USE_FULL_DATA)
print(head % "USE_KNN_VALID", USE_KNN_VALID)
print(head % "FINE_TUNE_EPOCH", FINE_TUNE_EPOCH)
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

