"""
基于 LangChain + Ollama 的轻量 RAG 示例（PDF → Chroma → 问答）。

与主工程中的 rag_pipeline / generation_pipeline 并存，用于快速验证或教学；
生产环境推荐走 config_center + rag_pipeline 路线。
依赖：pip install -U langchain-ollama cryptography>=3.1 sentence-transformers 等。
"""

import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ===================== 第一步：配置基础参数 =====================
PDF_PATH = "data/dji_manuals/DJI_Mavic_4_Pro_User_Manual_v1.2_zh-cn.pdf"  # 改成你的PDF路径
VECTOR_DB_PATH = "dji_chroma_db"
OLLAMA_MODEL = "deepseek-r1:7b"
EMBED_MODEL = "nomic-embed-text"  # 嵌入模型名称


# ===================== 第二步：加载并分割PDF =====================
def load_and_split_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")

    # 加载PDF
    try:
        print("🔄 尝试用PyPDFLoader加载PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ PyPDFLoader加载成功，共 {len(documents)} 页")
    except Exception as e:
        print(f"⚠️ PyPDFLoader加载失败：{str(e)}")
        try:
            print("🔄 尝试用UnstructuredPDFLoader加载PDF...")
            os.system("pip install unstructured pdf2image pillow -q")
            loader = UnstructuredPDFLoader(pdf_path)
            documents = loader.load()
            print(f"✅ UnstructuredPDFLoader加载成功，共 {len(documents)} 页")
        except Exception as e2:
            raise RuntimeError(f"所有PDF加载方法都失败：{str(e2)}")

    # 分割文本（移除encoding参数）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、"],
    )
    splitted_docs = text_splitter.split_documents(documents)
    print(f"✅ 文本分割完成，共 {len(splitted_docs)} 个文本块")
    return splitted_docs


# ===================== 第三步：构建向量库（修复嵌入模型） =====================
def build_vector_db(docs, db_path):
    """使用Ollama的nomic-embed-text嵌入模型构建向量库"""

    # 初始化Ollama嵌入模型（适配GPU）
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        num_gpu=1,  # 强制使用你的RTX 3050 Ti
        base_url="http://localhost:11434"  # Ollama默认地址，确保能访问
    )

    # 创建向量库目录
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"✅ 创建向量库目录：{db_path}")

    # 构建Chroma向量库
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    vector_db.persist()
    print(f"✅ 向量库构建完成，存储路径：{db_path}")
    return vector_db


# ===================== 第四步：构建RAG问答链 =====================
def build_rag_chain(vector_db):
    # 初始化LLM（GPU加速）
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_ctx=4096,
        num_gpu=1
    )

    # RAG提示词
    prompt = ChatPromptTemplate.from_template("""
你是大疆产品说明专业问答助手，仅根据提供的参考资料回答问题，不要编造信息。
如果参考资料中没有相关答案，请明确说明“未找到相关信息”。

参考资料：
{context}

用户问题：
{question}

请用简洁、准确的中文回答用户的问题：
    """)

    # 检索器
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 组装RAG链
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    print("✅ RAG问答链构建完成！")
    return rag_chain


# ===================== 第五步：问答函数 =====================
def dji_qa(question, rag_chain):
    print(f"\n📝 用户问题：{question}")
    print("🔍 正在检索相关手册内容...")
    try:
        answer = rag_chain.invoke(question)
        print(f"💡 回答：{answer}")
        return answer
    except Exception as e:
        print(f"❌ 出错了：{str(e)}")
        return None


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 1. 加载PDF
    try:
        docs = load_and_split_pdf(PDF_PATH)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        exit(1)

    # 2. 构建向量库
    vector_db = build_vector_db(docs, VECTOR_DB_PATH)

    # 3. 构建RAG链
    rag_chain = build_rag_chain(vector_db)

    # 4. 测试问答
    test_questions = [
        "大疆无人机Mavic_4_Pro相机使用的注意事项",
        "大疆无人机Mavic_4_Pro怎么充电？",
        "大疆无人机Mavic_4_Pro的遥控器怎么休眠？"
    ]
    for q in test_questions:
        dji_qa(q, rag_chain)