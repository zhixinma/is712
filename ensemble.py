from dataloader import read_label


fp1 = "submission_1.txt"
fp2 = "submission_2.txt"
fp3 = "submission_0.txt"
fp4 = "submission_4.txt"
fp = "submission.txt"

mean_1, var_1 = read_label(fp1)
for m, v in zip(mean_1, var_1):
    print(m, v)

# with open(fp1, "r") as f1:
#     res1 = f1.readlines()
#
#     with open(fp2, "r") as f2:
#         res2 = f2.readlines()
#
#         with open(fp3, "r") as f3:
#             res3 = f3.readlines()
#
#             # with open(fp4, "r") as f4:
#             #     res4 = f4.readlines()
#
#             with open(fp, 'w') as f:
#
#                 # for sp1, sp2, sp3, sp4 in zip(res1, res2, res3, res4):
#                 #     mean = [float(i.split("\t")[0]) for i in [sp1, sp2, sp3, sp4]]
#
#                 for sp1, sp2, sp3 in zip(res1, res2, res3):
#                     res = [[float(n) for n in i.split("\t")] for i in [sp1, sp2, sp3]]
#                     _mean, _var = zip(*res)
#                     mean = sum(_mean) / len(_mean)
#                     var = sum(_var) / len(_var)
#
#                     # std = 1.21255
#
#                     print(str(round(mean, 5))+'\t'+str(round(var, 5)))
#                     # f.write(str(round(mean, 5))+'\t'+str(round(var, 5))+"\n")
#



# from dataloader import read_label
# from const import *
# import os
#
# image_dir = sorted([file for file in os.listdir(train_dir) if "txt" not in file])
# train_y = read_label(train_dir)
# train_means, train_vars, _, _, _ = zip(*train_y)
#
# for i in range(len(image_dir)):
#     mean_i, var_i = train_means[i], train_vars[i]
#     src_img = image_dir[i]
#     tgt_mean_img = str(mean_i) + "_m_" + src_img
#     tgt_var_img = str(var_i) + "_v_" + src_img
#
#     cmd = "cp dataset/train/%s data_mean/%s" % (src_img, tgt_mean_img)
#     os.system(cmd)
#     cmd = "cp dataset/train/%s data_var/%s" % (src_img, tgt_var_img)
#     os.system(cmd)


# fp1 = "submission_var.txt"
# fp = "submission.txt"
#
# with open(fp1, "r") as f1:
#     res1 = f1.readlines()
#
#     with open(fp, 'w') as f:
#         for sp1 in res1:
#             _mean, _var = sp1.split("\t")
#             mean = 6.42169
#             var = float(_var)
#
#             # print(str(round(mean, 5)) + '\t' + str(round(var, 5)))
#             f.write(str(round(mean, 5))+'\t'+str(round(var, 5))+"\n")
#