# glm-rag-ecommerce-agent
基于LangChain + 智谱GLM 搭建的电商RAG智能Agent，支持知识库检索、工具调用、角色权限控制。

## ✨项目亮点
1. RAG检索增强：读取私有知识库，解决大模型幻觉问题，基于文档内容回答商品、售后相关问题
2. Agent工具调用：大模型自主判断，选择对应的工具（商品库查询）
3. 角色权限隔离：区分客户(customer)和商家(operator)身份
   - 客户：仅可查询商品介绍、售后质保信息，禁止生成营销文案
   - 商家：可调用工具生成电商营销文案
4. 结构化数据读取：支持读取JSON商品库 + TXT文档知识库

## 🛠️技术栈
Python、LangChain、智谱GLM Embedding & LLM、向量检索、工具调用、角色权限Prompt控制

## 📁项目结构

