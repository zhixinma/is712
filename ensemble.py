import numpy as np


def read_label(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    separator = "\t" if "\t" in lines[0] else " "
    y = [[float(e) for e in line.split(separator)] for line in lines]
    return y


def rmse(pred, gold):
    score = np.mean((pred - gold) ** 2) ** 0.5
    return score


def dist(mean_p, var_p, mean_g, var_g):
    d_m = rmse(mean_p, mean_g)
    d_v = rmse(var_p, var_g)
    return 0.6 * d_m + 0.4 * d_v, d_m, d_v


def ensemble(mean_s, var_s):
    return sum(mean_s) / len(mean_s), sum(var_s) / len(var_s)


def load_res(fps):
    ms, vs = [], []
    for fp in fps:
        m, v = zip(*read_label(fp))
        ms.append(np.array(m))
        vs.append(np.array(v))
        del m, v
    return ms, vs


def write(_mean, _var, tar):
    with open(tar, "w") as f:
        for m, v in zip(_mean, _var):
            line = "%.5f\t%.5f\n" % (m, v)
            f.write(line)


fp_f_res = "submission_test_resnet_lrs_0.txt"
fp_f_cnn = "submission_test_cnn_lrs_0.txt"
fp_f_res_cnn = "submission_test_resnet_cnn_lrs_0.txt"
fp_f_res_dis_reg = "submission_test_resnet_lrs_dis_reg_0.txt"
fp_f_res_val = "submission_test_resnet_lrs_val_0.txt"

fp_test_submission = "submission_test.txt"

fp_test = fp_sub
fp_final = gold
mean_f, var_f = zip(*read_label(fp_final))
mean_t, var_t = zip(*read_label(fp_test))
assert len(mean_f) == len(mean_t), (len(mean_f), len(mean_t))
d, d_m, d_v = dist(np.array(mean_t), np.array(var_t), np.array(mean_f), np.array(var_f))
print("%15s: %.5f, %.5f, %.5f" % ("submission(848)", d, d_m, d_v))

tag = ""
fp_list = [fp_f_res, fp_f_cnn, fp_f_res_cnn, fp_f_res_dis_reg, fp_f_res_val]
mean_list, var_list = load_res(fp_list)
mean, var = ensemble(mean_list, var_list)
d, d_m, d_v = dist(np.array(mean), np.array(var), np.array(mean_f), np.array(var_f))
print("%15s: %.5f, %.5f, %.5f" % (tag, d, d_m, d_v))
write(mean, var, "./submission_test.txt")










