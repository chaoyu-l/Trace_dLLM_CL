# Copyright (c) Microsoft Corporation.                      # 版权声明：代码版权归 Microsoft Corporation 所有
# SPDX-License-Identifier: Apache-2.0                       # 许可证声明：使用 Apache-2.0 开源许可证

# DeepSpeed Team                                            # 说明：该代码风格/结构来自 DeepSpeed 团队的工程

from datasets import load_dataset, load_from_disk           # 从 HuggingFace datasets 导入：加载在线数据集 / 从本地磁盘加载
from torch.utils.data import Subset                         # 从 PyTorch 导入 Subset：用于对数据集做子集切分（本文件暂未用到）
import re                                                   # 导入正则表达式库 re（本文件暂未用到）
import os                                                   # 导入操作系统接口 os：用于路径存在性检查等


# 只保留 prompt 和 train answer                              # 注释：该文件的数据接口设计目标：只抽取 prompt 和 answer


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
# 注释：这是一个“模板类”，用于统一不同数据集的接口（API）与格式（prompt/answer）
class PromptRawDataset:                             # 定义基础数据集类（父类），其它数据集类继承它

    def __init__(self, output_path, seed, local_rank, dataset_name):  # 初始化：传入输出目录、随机种子、分布式 rank、数据集名/路径
        self.output_path = output_path                       # 保存输出路径（例如缓存/预处理后的保存位置）
        self.seed = seed                                     # 保存随机种子（用于可复现的数据处理/采样）
        self.local_rank = local_rank                         # 保存当前进程的本地 rank（分布式训练时常用）
        # default load from disk                              # 注释：默认策略——某些数据集直接从磁盘加载
        if "Anthropic/hh-rlhf" in dataset_name:              # 如果 dataset_name 字符串包含指定数据集标识
            self.raw_datasets = load_from_disk(dataset_name) # 则将 dataset_name 当作“磁盘路径”，从本地磁盘读取数据集对象

    def get_train_data(self):                                # 获取训练集的接口（子类应覆盖实现）
        return                                               # 这里是占位实现：直接返回 None（表示未实现）

    def get_eval_data(self):                                 # 获取验证/评估集的接口（子类应覆盖实现）
        return                                               # 占位实现：返回 None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    # 注释：约定 prompt 格式类似：Human: ... Assistant: ...
    def get_prompt(self, sample):                            # 输入一条样本 sample，返回 prompt 文本（子类应覆盖）
        return                                               # 占位实现：返回 None

    # The chosen response should be in the format of: " " + actual_response_sentence
    # 注释：约定 answer/response 一般是 assistant 的回复文本
    def get_answer(self, sample):                            # 输入一条样本 sample，返回 answer 文本（子类应覆盖）
        return                                               # 占位实现：返回 None

    def get_prompt_and_answer(self, sample):                 # 输入样本 sample，返回拼接后的 prompt+answer（子类应覆盖）
        return                                               # 占位实现：返回 None



# added for Anthropic/hh-rlhf                                # 注释：下面是针对 Anthropic/hh-rlhf 数据集的适配实现
# adapt for llama2 chat                                      # 注释：适配 llama2-chat 风格的对话格式
"""
Data examples:                                               # 文档字符串：这里给出该数据集样例字段结构
{
    "chose": " Human: What are some cuss words in english? Assistant: Here’s an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: I haven't even thought about it.",
    "rejected": " Human: What are some cuss words in english? Assistant: Here’s an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: Ass."
}

"""                                                         # 结束文档字符串：仅用于说明数据格式，不影响运行

class AnthropichhrlhfDataset(PromptRawDataset):              # 定义 Anthropic/hh-rlhf 数据集适配类，继承 PromptRawDataset
    def __init__(self, output_path, seed, local_rank, dataset_name):  # 初始化：同父类参数
        super().__init__(output_path, seed, local_rank, dataset_name) # 调用父类初始化：保存路径/seed/rank，并按需 load_from_disk

        self.dataset_name = "Anthropic/hh-rlhf"              # 规范化记录该数据集的“标准名字”
        self.dataset_name_clean = "Anthropic_hh_rlhf"        # 记录一个“clean”版本名字（通常用于文件名/目录名避免特殊字符）

    def get_train_data(self):                                # 覆盖：返回训练集 split
        return self.raw_datasets["train"]                    # 从 raw_datasets 中取出 "train" split 并返回

    def get_eval_data(self):                                 # 覆盖：返回评估/测试集 split
        return self.raw_datasets["test"]                     # 此处将 "test" split 作为 eval 使用并返回

    # Human 和 Assitant 保持原样，相信模型！                 # 注释：保持原始文本中的 Human/Assistant 标记，不做替换/清洗
    def get_prompt(self, sample):                            # 覆盖：从样本中解析 prompt（对话上下文）
        segments = sample['rejected'].split('Assistant:')    # 用 "Assistant:" 作为分隔符切分 rejected 字符串（得到多段）
        prompt = "Assistant:".join(segments[:-1])            # 将除最后一段之外的内容重新用 "Assistant:" 拼回去（保留到最后一次回答之前）
        return prompt + "Assistant:"                         # 返回 prompt，并在末尾补上 "Assistant:" 作为模型要继续生成的位置

    def get_answer(self, sample):                            # 覆盖：从样本中解析 answer（这里取 rejected 的最后一段作为回答）
        segments = sample['rejected'].split('Assistant:')    # 同样按 "Assistant:" 切分
        rejected = segments[-1]                              # 取最后一段：即最后一个 "Assistant:" 之后的文本（最终回答内容）
        return rejected                                      # 返回该回答文本（注意：可能包含前导空格）

    def get_prompt_and_answer(self, sample):                 # 覆盖：直接返回完整的 prompt+answer（这里选 rejected 全串）
        return sample['rejected']                            # 返回 rejected 字段原文（包含 Human/Assistant 多轮对话）



class LocalJsonFileDataset(PromptRawDataset):                # 定义本地 JSON 文件数据集类（读取 train/eval/test.json）
                                                             # 注意：虽然继承 PromptRawDataset，但这里自己用 load_dataset('json') 加载

    def __init__(self, output_path, seed, local_rank, dataset_name, for_backbone=False):  # 初始化：多一个 for_backbone 控制开关
        super().__init__(output_path, seed, local_rank, dataset_name) # 调用父类初始化（父类可能会根据 dataset_name 决定 load_from_disk）
        self.dataset_name = "local_jsonfile"                 # 设置数据集名字标识：本地 json 文件
        self.dataset_name_clean = "jsonfile"                 # 设置 clean 名字：用于路径/文件名更友好
        assert os.path.exists(dataset_name), f"Not found, plz check path {dataset_name}!"  # 检查 dataset_name 路径存在，否则直接报错
        self.for_backbone = for_backbone                     # 保存 for_backbone 参数：用于控制是否为 backbone 训练准备（本段代码未使用）
        self.raw_datasets = load_dataset('json',             # 使用 datasets.load_dataset 读取 JSON 数据集
                                         data_files={        # data_files 指定每个 split 对应的文件路径
                                             "train":        # 训练集 split 名称
                                             dataset_name + '/train.json',  # train split 文件：<dataset_name>/train.json
                                             "eval":         # 验证集 split 名称
                                             dataset_name + '/eval.json',   # eval split 文件：<dataset_name>/eval.json
                                             "test":         # 测试集 split 名称
                                             dataset_name + '/test.json',   # test split 文件：<dataset_name>/test.json
                                         })                  # 结束 load_dataset 调用：返回 DatasetDict（含 train/eval/test）

    def get_train_data(self):                                # 覆盖：获取训练集数据
        if self.raw_datasets['train'] is not None:           # 如果 train split 成功加载且不为空
            return self.raw_datasets['train']                # 返回 train split（datasets.Dataset 对象）
        return None                                          # 否则返回 None

    def get_eval_data(self):                                 # 覆盖：获取验证集数据
        if self.raw_datasets['eval'] is not None:            # 如果 eval split 存在
            return self.raw_datasets['eval']                 # 返回 eval split
        return None                                          # 否则返回 None

    def get_test_data(self):                                 # 额外提供：获取测试集数据（父类没声明但这里提供了）
        if self.raw_datasets['test'] is not None:            # 如果 test split 存在
            return self.raw_datasets['test']                 # 返回 test split
        return None                                          # 否则返回 None

    def get_prompt(self, sample):                            # 覆盖：从样本中读取 prompt 字段
        if sample['prompt'] is not None:                     # 若样本里 prompt 字段不为 None
            return sample['prompt']                          # 返回 prompt 文本
        return None                                          # 否则返回 None

    def get_answer(self, sample):                            # 覆盖：从样本中读取 answer 字段
        if sample['answer'] is not None:                     # 若样本里 answer 字段不为 None
            return sample['answer']                          # 返回 answer 文本
        return ''                                            # 若 answer 为 None，则返回空字符串（避免后续拼接/训练报错）

    def get_prompt_and_answer(self, sample):                 # 覆盖：返回 prompt 与 answer 的拼接版本
        if sample['prompt'] is not None and sample['answer'] is not None:  # 同时确保 prompt 和 answer 都存在
            return sample['prompt'] + "\n" + sample['answer']# 以换行符连接：prompt + \n + answer
        return None                                          # 若缺失任意字段，则返回 None