补跑原队列失败的 5 个 off-policy 实验。原始结果保留在 retry_plan.json 的 source。

先运行完整 256 回答/32 次更新的单步检查，通过后运行 200 步；失败继续后续批次，单次超时 4 小时。完成后自动删除 checkpoint。

数值修正：float32 log-softmax 和 importance ratios；指数前屏蔽 prompt/padding；非有限 loss/gradient 不进入 Adam，记录 skipped_nonfinite_updates（每轮对 32 次更新取均值）。因此补跑结果应单独标注，不能当作原始实现的重复运行。

GPU 错误本次未复现，8 张 GPU 的实际矩阵计算均通过；没有修改驱动。15 项回归测试通过。
