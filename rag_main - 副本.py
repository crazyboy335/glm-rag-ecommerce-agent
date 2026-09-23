import sqlite3
from langchain.tools import tool
from langchain_openai import ChatOpenAI
# ========== 数据库初始化（新增商品信息、销售订单表） ==========
def init_db():
    conn = sqlite3.connect("shop.db")
    cur = conn.cursor()
    # 库存表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS stock(
        goods_name TEXT PRIMARY KEY,
        num INTEGER
    )
    ''')
    # 商品信息表：价格、质保、优惠
    cur.execute('''
    CREATE TABLE IF NOT EXISTS goods_info(
        goods_name TEXT PRIMARY KEY,
        price REAL,
        warranty TEXT,
        promotion TEXT
    )
    ''')
    # 销售订单表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS orders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goods_name TEXT,
        sale_count INTEGER,
        sale_amount REAL
    )
    ''')
    # 初始化库存
    cur.execute("INSERT OR IGNORE INTO stock(goods_name,num) VALUES (?,?)", ("降噪耳机",28))
    cur.execute("INSERT OR IGNORE INTO stock(goods_name,num) VALUES (?,?)", ("有线耳机",56))
    cur.execute("INSERT OR IGNORE INTO stock(goods_name,num) VALUES (?,?)", ("蓝牙耳机",120))
    # 初始化商品详情
    cur.execute("INSERT OR IGNORE INTO goods_info(goods_name,price,warranty,promotion) VALUES (?,?,?,?)",
                ("降噪耳机",399,"1年整机质保，非人为损坏免费维修","下单立减50元"))
    cur.execute("INSERT OR IGNORE INTO goods_info(goods_name,price,warranty,promotion) VALUES (?,?,?,?)",
                ("有线耳机",79,"半年质保","两件8折"))
    cur.execute("INSERT OR IGNORE INTO goods_info(goods_name,price,warranty,promotion) VALUES (?,?,?,?)",
                ("蓝牙耳机",199,"1年质保","满199送收纳袋"))
    
    # 模拟销售数据
    cur.execute("INSERT OR IGNORE INTO orders(goods_name,sale_count,sale_amount) VALUES (?,?,?)",("蓝牙耳机",86,17114))
    cur.execute("INSERT OR IGNORE INTO orders(goods_name,sale_count,sale_amount) VALUES (?,?,?)",("降噪耳机",42,16758))
    cur.execute("INSERT OR IGNORE INTO orders(goods_name,sale_count,sale_amount) VALUES (?,?,?)",("有线耳机",35,2765))
    conn.commit()
    conn.close()
# ====================== 客户可用函数 ======================
def _query_stock_func(goods_name: str) -> str:
    """查询库存"""
    conn = sqlite3.connect("shop.db")
    cur = conn.cursor()
    cur.execute("SELECT num FROM stock WHERE goods_name = ?", (goods_name,))
    res = cur.fetchone()
    conn.close()
    if res:
        return f"{goods_name}当前库存：{res[0]}"
    return f"没有找到商品：{goods_name}"
def _query_goods_detail_func(goods_name: str) -> str:
    """查询商品价格、质保、优惠"""
    conn = sqlite3.connect("shop.db")
    cur = conn.cursor()
    cur.execute("SELECT price,warranty,promotion FROM goods_info WHERE goods_name = ?", (goods_name,))
    res = cur.fetchone()
    conn.close()
    if res:
        price,warranty,promotion = res
        return f"""【{goods_name}】
售价：{price}元
质保：{warranty}
活动：{promotion}"""
    return f"没有找到商品：{goods_name}"
# ====================== 商家可用函数 ======================
def _update_stock_func(goods_name: str, new_num: int) -> str:
    """修改库存，仅admin"""
    conn = sqlite3.connect("shop.db")
    cur = conn.cursor()
    cur.execute("REPLACE INTO stock(goods_name, num) VALUES (?, ?)", (goods_name, new_num))
    conn.commit()
    # 额外查询最新库存，返回确认信息，证据更充分
    cur.execute("SELECT num FROM stock WHERE goods_name = ?", (goods_name,))
    latest_num = cur.fetchone()[0]
    conn.close()
    return f"✅库存修改成功！商品【{goods_name}】当前最新库存：{latest_num}"

def _get_sale_stat_func() -> str:
    """商家：热销排行、总销量、总销售额"""
    conn = sqlite3.connect("shop.db")
    cur = conn.cursor()
    # 重点：GROUP BY goods_name，按商品聚合，同一个商品只输出一行
    cur.execute("""
        SELECT goods_name, SUM(sale_count) as sale_count, SUM(sale_amount) as sale_amount
        FROM orders
        GROUP BY goods_name
        ORDER BY sale_count DESC
    """)
    sale_list = cur.fetchall()

    # 汇总全部的总量、总金额
    cur.execute("SELECT SUM(sale_count), SUM(sale_amount) FROM orders")
    total_count, total_amount = cur.fetchone()
    conn.close()

    output = "====销售统计====\n"
    output += f"累计总销量：{total_count}件\n"
    output += f"累计总销售额：{total_amount}元\n\n热销排行：\n"
    for name, cnt, amt in sale_list:
        output += f"{name}：销量{cnt}件，销售额{amt}元\n"
    return output

# ------------------- Tool包装（只给大模型看描述） -------------------
@tool
def query_stock(goods_name: str) -> str:
    """查询商品库存。
    Args:
        goods_name:商品名称，可选：降噪耳机、有线耳机、蓝牙耳机
    """
    return _query_stock_func(goods_name)
@tool
def query_goods_detail(goods_name: str) -> str:
    """查询商品详情：价格、质保、优惠活动。客户咨询商品信息调用这个。
    Args:
        goods_name:商品名称，可选：降噪耳机、有线耳机、蓝牙耳机
    """
    return _query_goods_detail_func(goods_name)
@tool
def update_stock(goods_name: str, new_num: int) -> str:
    """修改商品库存，仅商家admin角色使用。
    Args:
        goods_name:商品名称
        new_num:新库存数量
    """
    return _update_stock_func(goods_name, new_num)
@tool
def get_sale_stat() -> str:
    """商家admin专用，获取销售报表：热销商品排行、总销量、总销售额"""
    return _get_sale_stat_func()
# ========== Agent入口 ==========
def agent_run(user_input: str, role: str):
    init_db()
    if role == "customer":
        system_prompt = """你是电商客服，面向客户。
客户可以查询：商品库存、商品价格/质保/优惠活动。
可选商品：降噪耳机、有线耳机、蓝牙耳机。

【规则】
不能查看销售报表，不能修改库存。
当你需要调用工具，但是缺少商品名称这类必要参数时，不要调用TOOL，直接追问用户。
例：用户只说“查库存”，你回复：请问你想查询哪一款商品的库存呢？

指令格式：
查库存 → TOOL:query_stock,goods_name=xxx
查商品详情（价格质保优惠） → TOOL:query_goods_detail,goods_name=xxx
不需要调用工具，直接自然语言回答。"""
    elif role == "admin":
        system_prompt = """你是店铺商家后台助手。
你可以：
1. 查询库存 TOOL:query_stock,goods_name=xxx
2. 修改库存 TOOL:update_stock,goods_name=xxx,new_num=数字
3. 查询商品详情 TOOL:query_goods_detail,goods_name=xxx
4. 查看销售报表（热销排行、总销量、销售额）TOOL:get_sale_stat

【规则】
调用工具前，如果缺少必填参数，不要生成TOOL指令，主动向用户询问补齐信息。
例：用户说“修改库存”，你回复：请问要修改哪个商品的库存，修改成多少数量？

不需要调用工具，直接自然语言回答。"""
    else:
        return "身份错误"
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        openai_api_key="deepseek密钥",
        openai_api_base="https://api.deepseek.com/v1"
    )
    resp = llm.invoke([
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":user_input}
    ])
    content = resp.content.strip()
    # 解析工具调用
    if content.startswith("TOOL:"):
        cmd = content.replace("TOOL:","")
        if cmd.startswith("query_stock,"):
            _, goods_str = cmd.split(",")
            goods_name = goods_str.split("=")[1]
            return _query_stock_func(goods_name)
        elif cmd.startswith("query_goods_detail,"):
            _, goods_str = cmd.split(",")
            goods_name = goods_str.split("=")[1]
            return _query_goods_detail_func(goods_name)
        elif cmd.startswith("update_stock,") and role == "admin":
            _, params_str = cmd.split(",",1)
            # params_str: goods_name=降噪耳机&num=100
            param_list = params_str.split("&")
            goods_name = param_list[0].split("=")[1]
            new_num = int(param_list[1].split("=")[1])
            return _update_stock_func(goods_name, new_num)
        elif cmd == "get_sale_stat" and role == "admin":
            return _get_sale_stat_func()
    
    return content
if __name__ == "__main__":
    # 本地测试
    print(agent_run("库存情况", "customer"))
    print("-"*30)
    print(agent_run("帮我看下销售报表，哪个卖的最好", "admin"))
