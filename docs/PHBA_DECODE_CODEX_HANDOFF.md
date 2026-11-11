# 给 Codex：PHBA cached decode 迁移与 GPU 验收

请实际完成迁移、验收、性能对比和测试命令交付，不要只给计划。

## 来源与范围

源仓库：`https://github.com/ReyJerry/flash-linear-attention`，分支 `phba-ab`。
消融基线 commit：`fcb30701a70ac0235ff24537fc63f16602d99e24`。
先锁定本次交付的新增 cached-decode commit，阅读它相对上述基线的 diff 和 `fla/ops/phba/DECODE.md`。

将增量迁移到当前已集成的 `akseli/fla`；统一实验、跨节点调度、工蜂和镜像流程放到现有 `gsa2`。先审计两个仓库真实路径、remote、依赖、正在运行的训练和现有 cache 接口。不要整仓覆盖，不修改正在运行的消融实验、checkpoint 或其他架构。只新增 inference-only 路径，不启动预训练。

## 已实现接口

`from fla.models.phba.inference import PHBADecoder`

`decoder = PHBADecoder(model.eval(), max_length=prompt_length+steps, stage="p1_split")`

随后调用 `decoder.prefill(input_ids)` 和 `decoder.decode(next_ids)`。返回 `[B,1,vocab]` logits。

这不是 Transformers Cache 接口，不要直接设置现有 `PHBAConfig(use_cache=True)` 或调用 `model.generate()`。原有接口故意保留不变。只支持 dense、等长 batch，不支持 packed decode、beam reorder 或 CUDA Graph。

四阶段：`p0`、`p1_state`、`p1_route`、`p1_split`，默认主质量模式应为 `phba_ablation="full"`。

P0 是边界舍入兼容的 eager 参考实现：每 token 重算当前部分块状态，并重新池化历史 centroid；不是已充分优化的常规 recurrent decode。不要将其加速比包装成相对 MoBA/Transformer 的优势。

## 必须保留的语义

1. Prefix 是当前 block 开始之前的历史。块内 query 不读取更新到当前位置的 recurrent state。
2. Prefill 尾块尚未并入 FP32 anchor；跨块时只能累计一次。缓存仅保留当前尾块 raw K/gate，历史仅存 RoPE K/V 和 snapshots。
3. Snapshots BF16/FP16，carry FP32；不得量化或减少 C/K/M 来获取更好结果。
4. 当前 token 先读取旧 prefix，再发布下一 block 的 snapshot。读取只允许 causal tokens。
5. Top-K 只在已完成历史 blocks 中选，K 包含 current block。Centroid 及 raw/rotary query 路径不混淆。
6. Split decode 合并 unnormalized softmax statistics；不是 separately-softmax 后平均。
7. 同分路由优先较小 block ID。检查 reduction 差异对 near-ties 的影响，不能静默改成近似 Top-K。
8. `current_state_only` / `routed_states_only` 仅用于专门消融；审稿人的主推理表必须使用 full PHBA。

## 已执行与未执行

源实现完成了阶段 CPU 测试：P0 8 项、延迟 state 后 9 项、centroid cache 后 10 项、split 后 12 项。另有 3 项执行真实 Triton kernel 源码函数体的 PyTorch CPU 语义模拟。最终 15 个 CPU test methods 通过，Python 语法检查通过。

没有实际执行 Triton 编译、CUDA kernels、完整 Transformers 模型集成、GPU 性能或 32K/64K 显存测量。不要把 CPU 通过当作 CUDA 验收。必须在目标环境实际执行 GPU 测试，发现错误先修复且不放宽容差。

## 测试顺序

先在目标 FLA 重新运行：

```bash
python tests/ops/test_phba_decode_cpu.py
python tests/ops/test_phba_decode_kernels_cpu.py
FLA_CI_ENV=0 python benchmarks/phba/run_decode_acceptance.py --output-dir /tmp/phba-decode-acceptance
```

Acceptance runner 会运行新增 GPU 算子/模型测试和原消融/PHBA 回归，检查新增 GPU 测试没有被 skip，并保存源码 digest。确认实际 `import fla; print(fla.__file__)` 指向目标仓库。

重点验收：FP16/BF16、batch 1/2、C-1/C/C+1、partial prefill tail、连续多个 block、空 prefix、Top-K tie、q/k norm、output gate、fused norm/MLP、cached logits 与完整 causal forward、旧训练路径和 checkpoint 参数不变。

随后使用真实训练 checkpoint 验证短序列。Benchmark 本身也带 checkpoint 短序列检查；失败不得绕过。

## 对比脚本

`benchmarks/phba/benchmark_decode.py`：四阶段全模型对比，生成原始 JSONL、CSV、Markdown，分别记录 prefill/decode peak allocated/reserved、每 token event 时间、wall latency、普通 token 和 block-boundary token、cache bytes。

`benchmarks/phba/benchmark_decode_components.py`：单层合成输入下，分别比较 eager/deferred state、重算/cached centroid routing、Torch/split readout。不能拿它代替端到端结果。

先跑 8K 小规模性能 smoke，再跑 32768/65536 + 256 incremental forward steps。主模型设置以真实 checkpoint 为准。Benchmark 的默认 continuation 是跨阶段一致的 teacher-forcing，`--greedy` 可做自由生成复核；明确说明这个口径。P95/P99 不足以涵盖每 128 token 一次的状态更新，必须查看 boundary/max 延迟。

两节点、每节点 8 卡：独立模型或候选各占一张空闲 GPU，并行但不干扰正在训练的任务。PHBA 四阶段可在一张独占 GPU 顺序测量，或在同型号不同卡测完后轮换复测。最终五模型主表需要另外调用 gsa2 已有 Transformer/MoBA/NHA/KDA 的真实 cached decode，不能复用 cache-free generation 计时。

## 合并、工蜂、镜像与交付

在目标工作分支迁移、提交并推送工蜂，遵循现有 Dockerfile/镜像 registry 流程构建镜像。保留完整源 commit、目标 FLA/gsa2 commit、镜像 digest、测试日志，不写入 token/SSH 私钥。

最终给出真实可复制命令：单卡 smoke、GPU acceptance、四阶段 32K/64K、component benchmark、双节点五模型主表、结果收集、镜像拉取/运行。先核实 checkpoint 路径、挂载、空闲卡号和 registry，不用猜测值或 `<placeholder>`。

报告逐项区分：已通过、失败、跳过、未执行。对于 GPU 编译/数值问题，先定位修复并重新验收；禁止降低 tolerance 或偷偷改模型。不要宣称必然加速或已经全面验证。
