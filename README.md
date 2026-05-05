# 🧠 智能客服问答系统（RAG + ReAct Agent扩展）

基于 LangChain 的企业级智能客服系统，支持用户上传 TXT 文档构建私有知识库，并提供 标准 RAG 链*与 ReAct Agent（自主调用工具）两种问答模式。  
具备对话历史持久化、MD5 去重、安全计算器、向量检索与生成一体化等功能。
---
✨ 主要功能
- 📄 **知识库管理**  
  - 上传 TXT 文件，自动分块、向量化并存入 Chroma 向量库  
  - MD5 去重机制，避免重复入库  
  - 支持增量更新  

- 💬 **双模式问答**  
  - **标准 RAG 链**：检索相关文档片段 + 历史对话 → 生成回答（无信息时拒绝回答）  
  - **ReAct Agent**：自主调用知识库检索工具 / 计算器工具，解决复杂问题（如“查询商品克重后计算总重量”）  

- 🗂️ **对话历史管理**  
  - 基于 `RunnableWithMessageHistory` + 文件存储  
  - 支持多会话隔离、历史截断（默认保留最近 30 条）、清空历史  

- 🔒 **安全与性能优化**  
  - 计算器工具使用 `simpleeval` 安全求值，避免代码注入  
  - Embedding 模型单例模式，减少 API 调用  
  - Session ID 路径安全过滤  
  - 日志记录与异常处理  

- 🎨 **交互界面**  
  - 基于Streamlit 构建的 Web 界面，支持服务模式切换、会话 ID 管理、实时聊天组件  
---

## 🛠️ 技术栈

| 类别             | 技术                                                                 
| ---------------- | ---------------------------------
| 框架             | Streamlit, LangChain                    
| 向量库           | Chroma (本地持久化)                     
| 大模型           | 通义千问 (qwen3-max)               
| Embedding        | text-embedding-v4 (DashScope)   
| Agent 框架       | LangGraph (create_react_agent)                  
| 存储             | 文件系统 (对话历史，MD5 记录)                 
| 安全计算         | simpleeval                    

---
## 📦 安装与运行
在pycharm终端输入：streamlit run app_qa.py 
### 1. 克隆仓库

```bash
git clone https://github.com/wjp263/Rag-
cd 智能客服系统
