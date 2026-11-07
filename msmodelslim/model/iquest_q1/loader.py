# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

# IQuest-Q1 是自定义架构：config.json 的 model_type="iquest_q1" 不在 transformers
# 的注册表里，AutoConfig.from_pretrained() 会直接抛 "does not recognize this
# architecture"。这里在 loader 被 import 时把配置类注册进 AutoConfig，
# 使 msmodelslim 侧读 config.json 走通（本模块由插件工厂懒加载，注册早于
# 适配器构造）。
from transformers import AutoConfig

from .configuration_iquest_q1 import IQuestQ1Config

try:
    AutoConfig.register(IQuestQ1Config.model_type, IQuestQ1Config)
except ValueError:
    # 已注册（同一进程内重复 import / 其它模块先注册过）
    pass


class IQuestQ1AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.iquest_q1.model_adapter:IQuestQ1ModelAdapter"
