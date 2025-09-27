import re
import json

def main(arg1: str) -> dict:
    # 默认返回值
    default_output = {
        "results": "",
        "ECHarts": "0",
        "chartType": "",
        "chartTitle": "",
        "chartData": "",
        "chartXAxis": ""
    }
    
    try:
        # 使用正则表达式提取被 ```json 和 ``` 包裹的内容
        match = re.search(r'```json\s*([\s\S]*?)\s*```', arg1)
        if not match:
            raise ValueError("输入字符串中未找到有效的 JSON 数据")
        
        # 提取 JSON 字符串
        json_str = match.group(1).strip()
        
        # 将 JSON 字符串解析为 Python 字典
        result_dict = json.loads(json_str)
    except Exception as e:
        # 如果解析失败，打印错误信息并返回默认输出
        print(f"解析失败: {e}")
        return default_output
    
    # 检查是否包含 ECHarts 字段
    if "ECHarts" not in result_dict:
        result_dict["ECHarts"] = "0"  # 默认设置为 "0"
    
    # 根据 ECHarts 的值动态检查图表相关字段
    if result_dict["ECHarts"] == "1":
        required_chart_fields = ["chartType", "chartTitle", "chartData", "chartXAxis"]
        for field in required_chart_fields:
            if field not in result_dict:
                result_dict[field] = ""  # 自动补全缺失字段为空字符串
    
    # 构造返回值
    return {
        "results": str(result_dict.get("results", "")),
        "ECHarts": str(result_dict.get("ECHarts", "0")),
        "chartType": str(result_dict.get("chartType", "")),
        "chartTitle": str(result_dict.get("chartTitle", "")),
        "chartData": str(result_dict.get("chartData", "")),
        "chartXAxis": str(result_dict.get("chartXAxis", ""))
    }
