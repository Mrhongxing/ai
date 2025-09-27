def main(arg1: dict) -> dict:
    # 提取输入字典的字段
    title = arg1.get("title", "")
    sql = arg1.get("sql", "")
    
    # 返回包含 title 和 sql 的字典
    return {
        "title": title,
        "sql": sql
    }
