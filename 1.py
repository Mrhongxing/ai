import re
import json

def main(arg1: str) -> dict:
    # 使用正则表达式提取被 ```json 和 ``` 包裹的内容
    match = re.search(r'```json\s*([\s\S]*?)\s*```', arg1)
    if not match:
        raise ValueError("输入字符串中未找到有效的 JSON 数据")
    
    # 提取 JSON 字符串
    json_str = match.group(1).strip()
    
    try:
        # 将 JSON 字符串解析为 Python 字典
        result_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}")
    
    # 返回包含解析结果的字典
    return {
        "result": result_dict,
    }