# Interview Walkthrough
## 1. 项目经验.
### 1.1 模型架构 + 训练步骤(10min)
VLA 架构:
输入: Video Stream + Prompt Instruction(Ego Status + Navigation)
Video Stream 经过 Vision Encoder(ViT)
Prompt 经过 Text Tokenizer
然后 Cat 到一起之后过 LLM.
取 LLM Last Hidden States 过 Action Head(5 层 Transfomer + MLP)

训练步骤:
VLM PreTrain: Knowledge Injection. QADataset.
- CausalLMLoss. 目的是学习自动驾驶中的路况场景, 轨迹以 Text 形式作为 GT.
SFT: 在 VLM 的基础上加上 Action Head, 以 Trajectory Point 的形式学习.
- SFT 冻结 vision_model, mm_project, lm_head.
- SmoothL1 Loss: 当 Loss 较小时, 接近 L2, 梯度更加平滑. 当 Loss 较大时, 接近 L1, 对异常值不敏感.
- lr 10^-4
- Action Head 的实现方式: Autoregressive OR Regression. 他们的优劣.
    - 多模态的角度: Autoregressive 具有多模态, Regression 有模态坍塌的风险.
    - 精度的角度, Autoregressive 每个轨迹点依赖于上一个轨迹点, 如果某个点误差较大, 会造成整条轨迹误差大.
RL: SFT 的本质是模仿学习, RL 可以突破数据集的限制, 可以 Cover 一些 Corner 的极端场景, 是量产上车的必要步骤.
- 只训练 Trajectory Head.
- GRPOLoss
- lr 10^-5

### 1.2 量产: SFT Prompt 设计, RL Reward 设计, 数据挖掘.(30min, 10min/topic)
Prompt: Prompt Processor + Text Processor.
Prompt Processor 输出带占位符的 Token 的模板字符串, Text Processor 负责数值填充 + Tokenization.
ChatML 格式: System + User(Permutable + Suggestion) + Assistant.
用更结构化, 语义化的 Prompt 代替隐式的数组位置编码. Key-Value 对更适合 LLM.

RL Safety Reward: Small Distance + Collision + Relief.
Other Reward: Traffic Light Reward + Road Boundary Reward + GT Reward.

数据挖掘: 规则 + 模型 + 向量 + 人工.
数据配平, 共 4000W Clip 数据, 其中 70% 大盘数据, 其他为各个场景数据, 比如变道/跟车/过十字路口/横穿行人

### 1.3 预研: Memory VLA(10min)
Perception Compression, Memory Retrieval, Memory Gate Fusion, Memory Consolidation.

适合长时序的场景. 比如行人被遮挡. 决策意图不稳定, 换道.

## 2. 基础知识.
Transformer: Encoder / Decoder / MultiHeadAttn / Positional Embedding.
Activation: ReLU(max(0, x)), Sigmoid: 1 / 1 + e^(-x)
Regularization: Drop, Stop Early
Optimization: SGD, AdamW

Reinforcement Learning: PPO & GRPO.
GRPO: KL, Advantage. No Value Model.
PPO: Actor Critic.

VLM: Qwen. Decoder-Only 架构. mm_project, lm_head. ViT
Decoder-Only 的优势:
- 与人类语言的使用方式更接近(Next Token Prediction)
- KV Cache, 运行更高效.
- 实验证明, 更适合 Scaling Law.

## 3. LeetCode.