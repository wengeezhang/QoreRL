# QoreRL 后续工作

本文件记录实现进度与后续工作。架构原则见 [design.md](design.md)。实验 recipe 的本地实现已落地，原生 GPU 基线和收益验收仍待进行；勾选的实现项不代表 GPU 验收通过。

## P0：原生 BF16 基线验收

- [ ] 在同一台双 NVIDIA GPU 机器运行 Megatron + vLLM，记录 GPU、CUDA、PyTorch、TE、Bridge、MCore、vLLM 版本和完整命令。
- [ ] 同 snapshot、同 token 序列比较 behavior/training log-prob，检查有限值与差异尾部；将参数更新引起的变化与训推数值差异分开。
- [ ] 用非零组内优势完成至少两次真实 optimizer update 和 refit，验证权重内容变化、安装覆盖及新权重的实际执行结果。发布版本递增不能替代此检查。
- [ ] 验证 EOS、长度截断、变长样本、不同 microbatch 大小的 loss/梯度归一化。
- [ ] 中断/恢复及故障注入：恢复 optimizer/进度/版本，确认半更新不会继续服务；实测 GPU 恢复可重复性。
- [ ] 生成独立验收报告。运行结束只记录 `native_gpu_executed`，不自动把它解释为验收通过。

验收入口和详细条件见 [开发指南](docs/ray-grpo.zh-CN.md)。当前 GPU 验证由用户后续安排。

## P1：第一个可用的原生低精度 recipe

已实现范围见 [MXFP8/Hadamard recipe](docs/recipes/mxfp8-hadamard.zh-CN.md)。GPU 收益验收的前提是 P0 通过、有可重复运行的 BF16 对照。

- [x] 实现限定 MLP down projection 的原生 MXFP8 训练/导出/rollout 路径，以及固定 Hadamard 对照；明确 Fprop、Dgrad、Wgrad 精度，其他层保持 BF16。
- [x] 定义首组 recipe 规格、包含选项的身份、固定 signs/target 状态及同配置恢复检查。未来动态 scales/history 仍需显式扩展。
- [x] 在启动时验证首组 recipe 的模型、形状、硬件/后端支持条件，不静默 fallback。
- [ ] 在 GPU 跑算子 probe 和三组 GRPO 配置，确认非零实际更新、refit 和恢复；检查 native quantizer 之间的数值差异。
- [ ] 为每个 recipe 给出来源、复现配置、测试工作负载、数值结果、训练质量、显存与端到端成本；报告无收益和失败的情况。
- [x] 保留 canonical 权重，refit 后重建 vLLM 旋转/量化/kernel layout 缓存，失败时不确认版本；实际 GPU 执行结果检查仍待完成。
- [ ] 在实测有收益后优化 FWHT CUDA kernel、重复权重变换、缓存显存和同步开销，再扩展 MoE/并行。

验收：在同一真实工作负载上能复现 BF16 与 MXFP8 对照，并且用户能确认实际生效的配置与已知限制。之后逐个引入误差缓解方法，组合需单独验证。

## P1：训练内 recipe 观测

当前入口与测量边界见 [Recipe Insights](docs/recipe-insights.zh-CN.md)。实现完成不代表原生 GPU 数据已验证。

- [x] 增加默认关闭的观测配置：默认每 20 步轮转两层，限制输入行数和 canonical 副本字节数；使用真实更新前 X、W 配对重放三种前向方案，记录版本与失败状态。
- [x] 提供实时层视图、独立 HTML 报告和显式标记的合成演示；诊断配置可在恢复时调整，不改变训练 recipe。
- [ ] 在原生 GPU 验证采样、CUDA event 计时、误差、随机数/参数/梯度不变性，以及启用前后的训练结果；单独测观测对总耗时和显存峰值的影响，不能用 `probe_seconds` 或副本字节预算代替。
- [ ] 按实际问题补充 Dgrad/Wgrad 和梯度方向 probe，显式计入上游梯度与激活保留成本；保持配对参考，不把局部代理指标视为训练质量。
- [ ] 补充同 snapshot、同 token 的训推 log-prob 尾部，以及端到端质量/成本对照；定义跨窗口比较和覆盖规则，避免将不同 step 的层观测当成一次全模型测量。

## P1：观测、暂停、用户调整和恢复

触发条件：已有真实 recipe 和可解释的故障/退化观测。

- [ ] 区分数值非法状态、统计异常和训练质量退化。记录原始证据、可能原因与支持的调整选项，不把 outlier 或负梯度余弦直接判为根因。
- [ ] 配置 warning/pause 条件与安全停止边界；保留最近完整恢复点，同时保存 incident，不用异常状态覆盖恢复点。
- [ ] 实现修改配置后的兼容性检查和显式迁移：决定哪些模型/optimizer/recipe 状态保留、重置或重建。
- [ ] 增加数据游标、scheduler、随机 recipe 等能力时同步补齐 checkpoint。恢复协议记录 schema、配置与延续来源。
- [ ] 前向方案变化时协调训练与 rollout，清空旧在途数据，重新发布并采样。不要直接放开当前严格的同配置检查。

验收：用户能够从告警证据选择一个受支持的改动，恢复后确认实际生效，并保留改动前后质量与成本对照。暂不要求自动选择或热切换。

## P2：由真实扩展需求驱动的执行与接口改进

| 待办 | 何时做 | 验收要点 |
| --- | --- | --- |
| 原生多 rank learner | 单 rank GPU 基线通过，开始使用多卡模型/数据并行 | 使用 Megatron 原生 optimizer、schedule、梯度同步和分片 checkpoint；不要强制继承单 rank 更新循环 |
| 可替换的策略目标 | 引入第二个真实算法或 correction recipe | 将目标选择与 backend 执行分开，保留梯度、mask 和 loss 聚合语义；无需先做通用插件平台 |
| token log-prob / fused loss 接口 | 词表并行或 logits 内存成为瓶颈 | 避免强制 gather 完整 `[B,S,V]` logits，验证并行 loss 与梯度的一致性 |
| 分片/分桶/GPU 权重传输 | 全量 CPU 导出成为内存或同步瓶颈 | 区分版本元数据与 payload/传输 handle，完整安装后确认版本；失败不继续服务 |
| 明确任务与奖励选择 | 添加第二个真实任务/数据集 | 解除 backend 与 task 的默认绑定，支持独立 scorer 和数据状态恢复 |
| dynamic batch 与 sequence packing | padding 或变长负载显著影响成本 | 保持回答/有效 token 的聚合定义，检查边界 mask、重排和梯度累积 |
| 缓存、CUDA graph 与性能优化 | 原生正确性基线成立且有 profile | refit 后缓存/图引用有效，数值与成本对照充分，不能用配置成功代替有效执行 |
| 异步采样与训练 | 同步等待成为测得的主要瓶颈 | 解耦发布版本与 batch step，定义陈旧度/correction/丢弃策略，保留实际行为概率和版本 |
| 更独立的 Ray 资源/生命周期模块 | 第二种真实部署/调度方式出现 | 主循环保持可读；不提前拆出没有独立职责的多层 controller/service |

上述扩展可以逐步替换执行实现，样本身份、behavior log-prob、recipe 身份、发布确认和恢复边界应保持明确。必要的协议演进使用 schema 和迁移，而非声称当前接口已经适配全部未来场景。

## 长期研究：从候选比较到自适应

- [ ] 在原生执行路径重新评估 Wgrad BF16 激活、选择性更高精度、缩放、旋转及组合；计入保留激活、校准、变换和同步成本。
- [ ] 用同 snapshot、同样本的配对实验区分干预与分布变化，分开候选筛选数据和接受评估数据。
- [ ] 验证梯度代理指标是否能预测实际学习收益；不寻找一个未经证实的通用“安全误差阈值”。
- [ ] 根据可复现证据推荐受支持的调整，由用户决定；再研究有预算和条件约束的自动动作、复测与恢复。

[旧 tiny-MoE 实验](docs/experiments/2026-11-08/README.md) 保留了“负余弦不保证修复收益”的历史证据，不是已验证 recipe，也不是重新采用旧默认校准策略的依据。
