# LEGACY: Local API connectivity test; not a primary pipeline entry point.
import os
from openai import OpenAI

# 确保你的DeepSeek API Key已通过环境变量 DEEPSEEK_API_KEY 设置
# client = OpenAI(
#     api_key=os.environ.get("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com/v1"
# )

try:
    print("正在测试DeepSeek API连通性与账户状态...")
    # 使用一个极短、低成本的提示来测试
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hi"}],  # 仅发送一个字，成本最低
        max_tokens=5,  # 限制返回长度，进一步降低成本
        timeout=15
    )
    print("✅ **API连通性测试成功！**")
    print(f"   模型回复: {response.choices[0].message.content}")
    print("   这说明：1. 网络畅通；2. API Key有效；3. 账户有充足余额。")

except Exception as e:
    print(f"❌ 请求失败。")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {e}")

    # 根据错误信息给出明确指引
    if "402" in str(e) or "Insufficient Balance" in str(e):
        print("\n👉 **核心问题：账户余额不足。**")
        print("   请立即：")
        print("   1. 访问 https://platform.deepseek.com")
        print("   2. 登录后，检查并『领取免费额度』或进行『充值』。")
        print("   3. 确保充值后余额大于0。")
    elif "401" in str(e) or "Authentication" in str(e):
        print("\n👉 问题：API Key无效或未设置。")
        print("   请检查环境变量 DEEPSEEK_API_KEY 是否设置正确。")
    elif "timeout" in str(e).lower():
        print("\n👉 问题：网络连接超时。")
        print("   请确认已清除代理（http_proxy/https_proxy环境变量）。")