from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")

# 模型初始化
llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=api_key
)

# 加载知识库
loader = TextLoader("test.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 工具定义
@tool
def search_knowledge_base(query: str) -> str:
    """查询本地知识库，获取RAG、Agent、Chroma、LangChain相关资料。用户询问相关概念时使用。"""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

@tool
def calculator(expression: str) -> str:
    """数学计算，用户有加减乘除等数学运算需求时调用。输入数学表达式字符串。"""
    return str(eval(expression))
    
@tool
def query_goods(goods_name: str) -> str:
    """
    查询商品信息工具，用户询问商品参数、质保、运费时调用
    :param goods_name: 用户要查询的商品名称
    """
    with open("goods.json", "r", encoding="utf-8") as f:
        goods_list = json.load(f)
    # 简单匹配商品
    for goods in goods_list:
        if goods_name in goods["name"]:
            return json.dumps(goods, ensure_ascii=False)
    return "未找到该商品信息"
    
@tool
def generate_product_copy(product_info: str) -> str:
    """
    电商营销文案生成工具，用户需要商品标题、营销短句、商品简介时调用
    :param product_info: 商品的基础信息描述
    """
    prompt = f"""
    根据下面的商品信息，生成适合电商平台的简短营销文案，风格简洁有吸引力。
    商品信息：{product_info}
    """
    resp = llm.invoke(prompt)
    return resp.content



def agent_run(user_query: str, user_role: str):
    """
    :param user_query: 用户提问内容
    :param user_role: 用户身份，可选：customer（买家） / operator（商家运营）
    """
    # 根据身份分配可用工具
    if user_role == "customer":
        available_tools = [search_knowledge_base, query_goods]
    elif user_role == "operator":
        available_tools = [search_knowledge_base, query_goods, calculator, generate_product_copy]
    else:
        available_tools = [search_knowledge_base, query_goods]

    # 系统提示词：角色业务约束
    system_prompt = """
你是电商商品咨询助手。
规则：
1. 如果当前用户身份是customer（买家）：只能回答商品查询、知识库里面的商品信息。严禁生成营销文案、广告宣传文案。
当买家要求写宣传文案、带货文案、销售文案时，统一回复：抱歉，我只能为您介绍商品相关信息，无法生成营销文案。
2. 如果当前用户身份是operator（商家运营）：可以使用全部授权工具，允许调用工具生成商品营销文案。
如果用户询问商品名称、价格、参数，优先调用query_goods工具查询商品资料。
"""

    llm_with_tools = llm.bind_tools(available_tools)
    # 传入系统提示词 + 用户提问
    response = llm_with_tools.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ])

    # 判断是否调用工具
    if response.tool_calls:
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"\n【Agent自主调用工具】工具名称：{name}")
            print(f"工具入参：{args}")
            tool_result = ""
            if name == "search_knowledge_base":
                tool_result = search_knowledge_base.invoke(args["query"])
            elif name == "query_goods":
                tool_result = query_goods.invoke(args["goods_name"])
            elif name == "calculator":
                tool_result = calculator.invoke(args["expression"])
            elif name == "generate_product_copy":
                tool_result = generate_product_copy.invoke(args["product_info"])
            else:
                tool_result = "未知工具"

            final_resp = llm_with_tools.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    {"role": "tool", "tool_call_id": tool_call["id"], "content": tool_result}
                ]
            )
            return final_resp.content
    else:
        return response.content








if __name__ == "__main__":
    print("=====电商Agent测试程序=====")
    print("身份可选: customer(客户), operator(商家运营)")
    role = input("请输入你的身份: ")
    
    # =========新增的身份校验代码=========
    while role not in ["customer", "operator"]:
        print("输入错误！身份只能选 customer 或者 operator，请重新输入")
        role = input("请输入你的身份: ")
    # ==================================================
    
    while True:
        question = input("\n请输入提问: ")
        if question == "exit":
            break
        res = agent_run(question, role)
        print("Agent回复: ", res)


