from typing import List, Union
from pydantic import BaseModel, ValidationError, field_validator
import re
from typing import Optional, Literal
from notify import notify_fail
from opencompass.utils import extract_non_reasoning_content
from opencompass.models import (
    HuggingFacewithChatTemplate,
    TurboMindModelwithChatTemplate,
    TurboMindModel
)
import ast


# dict数据处理为kv用于后续校验
def parse_config_line(line):
    # 分割键和值
    key, value_str = line.split('=', 1)
    key = key.strip()
    value_str = value_str.strip()

    # 处理字典类型的值
    if value_str.startswith('dict(') and value_str.endswith(')'):
        # 提取括号内的内容并转换格式
        inner = value_str[5:-1].strip()
        formatted = re.sub(r'\b([a-zA-Z_]\w*)\s*=\s*', r"'\1': ", inner)
        dict_str = '{' + formatted + '}'
        print(formatted)
        try:
            value = ast.literal_eval(dict_str)
        except (SyntaxError, ValueError):
            value = value_str  # 解析失败时保留原字符串
    else:
        value = value_str  # 非字典值直接保留

    return {key: value}


class InputModel(BaseModel):
    eval_owner: str
    model_path: str
    gpu_num: Literal[1, 2, 4, 8]
    fullbench_version: str
    eval_type: str
    subdataset: Optional[Union[List[str], str]] = None
    docker_image: str
    conda_env: str
    backend_type: str
    infer_worker_nums: str
    feishu_token: str
    dlc_config: str
    workspace_id: str
    data_source_id: List
    model_config_test: object

    @field_validator('model_path')
    def validate_model_structure(cls, value: str) -> str:
        """
        验证多行逗号分隔格式：
        - 每行必须有且只有一个逗号
        - 每行分割后必须得到两个非空元素
        - 允许前后空格但会自动清除
        - 空行会自动跳过
        """
        if not value:
            raise ValueError("输入不能为空")

        # 统一换行符并过滤空行
        lines = [
            line.strip()
            for line in value.replace('\r', '').split('\n')
            if line.strip()  # 跳过空行
        ]

        for line_num, raw_line in enumerate(lines, start=1):
            # 检查逗号数量
            if raw_line.count(',') != 1:
                raise ValueError(
                    f"第 {line_num} 行格式错误 -> '{raw_line}'\n"
                    f"必须包含且仅包含一个逗号"
                )

            # 分割并校验元素
            parts = [part.strip() for part in raw_line.split(',')]
            if len(parts) != 2:
                raise ValueError(
                    f"第 {line_num} 行分割异常 -> '{raw_line}'\n"
                    f"分割后应得到2个元素，实际得到 {len(parts)} 个"
                )

            if not all(parts):
                raise ValueError(
                    f"第 {line_num} 行存在空值 -> '{raw_line}'\n"
                    f"两个元素均不能为空"
                )

        return value

    @property
    def parsed_pairs(self) -> list[tuple[str, str]]:
        """获取结构化数据"""
        return [
            tuple(line.strip().split(','))
            for line in self.model_path.split('\n')
            if line.strip()
        ]
    
    @field_validator('subdataset')
    def validate_subdataset(cls, value):
        if value is None:
            return value  # 允许直接为 None
        
        if isinstance(value, str):
            # 检查是否以 [ 开头和 ] 结尾
            if not (value.startswith("[") and value.endswith("]")):
                raise ValueError("入参必须用 [ ] 括起来的形式，如[*race_datasets]")
            
            stripped = value[1:-1].strip()  # 移除方括号
            parsed_items = [item.strip() for item in stripped.split(",") if item.strip()]
        else:
            parsed_items = value
            
        for item in parsed_items:
            # 检查每个元素是否以 * 开头且长度至少为 2（如 "*xxx"）
            if not (item.startswith('*') and len(item) >= 2):
                raise ValueError(f"数据集 '{item}' 不正确: 每个数据集需要以 '*' 开头且应该包含datasets字段 (如: '*race_datasets')")
        return value


def parse_value(value_str):
    value_str = value_str.strip()
    
    # 使用正则表达式匹配以dict(开头，以)结尾的结构，允许内部有任意字符（包括空格）
    if re.fullmatch(r'dict\s*\(.*\)', value_str, re.DOTALL):
        try:
            # 安全的环境，仅允许dict函数
            env = {'dict': dict}
            value = eval(value_str, env)
            if isinstance(value, dict):
                return value
        except Exception as e:
            # 解析失败，返回原始字符串
            print(e)
            pass
    # 非dict结构或解析失败，返回原始字符串
    return value_str


def parse_value_new(value_str):
    # 处理 dict(...) 格式的值
    if value_str.startswith('dict(') and value_str.endswith(')'):
        inner = value_str[5:-1].strip()
        parsed_dict = {}
        # 分割内部键值对
        for part in inner.split(','):
            part = part.strip()
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip()
                try:
                    # 尝试将值解析为 Python 字面量（如数字、布尔值）
                    parsed_v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    # 解析失败则保留原始字符串（如未定义的变量名）
                    parsed_v = v
                parsed_dict[k] = parsed_v
        return parsed_dict
    else:
        # 非字典值：尝试解析或保留原始字符串
        try:
            parsed_value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            parsed_value = value_str
        return parsed_value


def parse_config_file(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 跳过空行和注释行
            # 分割键和值，只分割第一个等号
            parts = line.split('=', 1)
            if len(parts) != 2:
                config[key] = None
                continue  # 报错！！
            key, value_str = parts
            key = key.strip()
            value = parse_value_new(value_str.strip().replace("\\n", "\n"))
            config[key] = value

    try:
        # 转换并验证数据
        validated = InputModel(**config)
        print("✅ 验证成功:")
    except ValidationError as e:
        print(f"❌ 数据验证失败:")
        fail_content = ''
        for error in e.errors():
            loc = "->".join(map(str, error['loc']))
            fail_content += f"字段 {loc}: {error['msg']}\n"
            print(f"字段 {loc}: {error['msg']}")
        #notify_fail('❌ 数据验证失败', fail_content, config["feishu_token"], 123, 'jenkins_url')
    except Exception as e:
        print(f"❌ 未知异常: {e}")
        #notify_fail('❌ 数据验证异常', "❌ 数据验证异常", config["feishu_token"], 123, 'jenkins_url')
    

# 示例使用
if __name__ == "__main__":
    parse_config_file('/home/zhulin1/code/lmdeployLin/params.txt')




