import json
from rag_main import agent_run
import requests

# 配置你的大模型API，用于LLM-as-Judge
# 注意：这里用和你 rag_main.py 里一样的模型名和key
DEEPSEEK_API_KEY = "你的密钥"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
JUDGE_MODEL = "deepseek-v4-flash"  # 改成和你 rag_main.py 里一样的模型

def llm_judge(query, role, agent_resp, expect):
    judge_prompt = f"""
你是评测裁判，需要判断智能体回答是否符合预期。
用户角色：{role}
用户提问：{query}
智能体输出：{agent_resp}
预期行为：{expect}

输出JSON，只有两个字段：
score：0~1，1完全符合预期，0完全不符合
reason：一句话理由
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0
    }
    
    res = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
    result = res.json()
    
    # 先打印原始返回，方便调试
    print(f"[调试] API返回: {result}")
    
    if "choices" not in result:
        print(f"[错误] API返回异常: {result}")
        return {"score": 0, "reason": f"API调用失败: {result.get('error', {}).get('message', '未知错误')}"}
    
    content = result["choices"][0]["message"]["content"]
    
    # 尝试解析JSON，如果失败就直接用文字判断
    try:
        return json.loads(content)
    except:
        # 如果模型没输出标准JSON，就手动判断
        return {"score": 0.5, "reason": f"模型输出无法解析: {content[:100]}"}

def run_eval():
    with open("test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    report = []
    pass_count = 0
    
    for case in cases:
        print(f"\n====Case {case['case_id']} 【{case['category']}】====")
        print(f"Query: {case['query']}, Role: {case['role']}")
        
        resp = agent_run(case["query"], case["role"])
        print(f"Agent输出：{resp}")
        
        judge_result = llm_judge(case["query"], case["role"], resp, case["expect"])
        score = judge_result["score"]
        reason = judge_result["reason"]
        
        if score >= 0.8:
            pass_count += 1
            
        report.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "role": case["role"],
            "agent_response": resp,
            "expect": case["expect"],
            "score": score,
            "reason": reason
        })
        print(f"评测得分：{score}, 原因: {reason}")
    
    total = len(cases)
    pass_rate = pass_count / total
    
    print(f"\n====评测汇总报告====")
    print(f"总用例：{total}，通过用例：{pass_count}，通过率：{pass_rate:.2%}")
    
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("评测报告已保存 eval_report.json")

if __name__ == "__main__":
    run_eval()
