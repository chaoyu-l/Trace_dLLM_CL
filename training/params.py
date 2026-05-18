# training/params.py

# 1. 只保留这两个核心模块的导入
from model.base_model import CL_Base_Model
from model.lora import lora

# 2. 精简映射字典，只保留用得上的 keys
Method2Class = {
    "base": CL_Base_Model,   # 对应：全参数微调 (Full Fine-tuning)
    "LLaDA": CL_Base_Model,  # 对应：LLaDA 全参数 (本质同上，只是别名)
    "lora": lora             # 对应：LoRA 微调
}

# 3. 数据集列表保持不变 (这是你的实验任务序列)
AllDatasetName = [
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten"
]

# SSR (ACL 2024) 基准的 10 个 SuperNI 任务。顺序 = SSR 官方 Table 2 的呈现顺序。
# qa/qg/sa/sum/trans/dsg/expl/para/pe/pos 分别对应：
#   Cosmos QA / SQuAD-QG / Amazon SA / Reddit Sum / TED 翻译 /
#   AirDialogue DSG / HotpotQA Expl / ParaNMT Para / 列表切片 PE / POS 分类。
AllDatasetName_SSR = [
    "qa",
    "qg",
    "sa",
    "sum",
    "trans",
    "dsg",
    "expl",
    "para",
    "pe",
    "pos",
]