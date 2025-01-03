from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 初始化 Flask 应用
app = Flask(__name__)

# 加载本地大模型
def load_model():
    mode_path = 'G:/Pycode/Meta-Llama-3___1-8B-Instruct'
    lora_path = 'G:/Pycode/self-llm-master/examples/Chat-嬛嬛/output/llama3_1_instruct_lora/checkpoint-699'

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    return tokenizer, model

tokenizer, model = load_model()

# 生成回复的逻辑
def generate_response(prompt):
    messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt")

    # 确保model_inputs转移到与模型相同的设备上
    model_inputs = model_inputs.to(model.device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 主页
@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chat with 甄嬛</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {
                    background-color: #f8f9fa;
                }
                .chat-container {
                    max-width: 800px;
                    margin: 50px auto;
                    background: #fff;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                }
                .chat-log {
                    height: 400px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                }
                .message {
                    margin-bottom: 10px;
                }
                .user-message {
                    text-align: right;
                }
                .user-message .content {
                    background-color: #007bff;
                    color: white;
                    border-radius: 10px;
                    padding: 8px 12px;
                    display: inline-block;
                    max-width: 70%;
                }
                .bot-message .content {
                    background-color: #e9ecef;
                    color: black;
                    border-radius: 10px;
                    padding: 8px 12px;
                    display: inline-block;
                    max-width: 70%;
                }
                .input-group {
                    margin-top: 20px;
                }
                .loading {
                    display: none;
                    color: #6c757d;
                    font-style: italic;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <div class="chat-container">
                <h1 class="text-center mb-4">Chat with 甄嬛</h1>
                <div class="chat-log" id="chat-log">
                    <!-- 聊天记录会动态加载到这里 -->
                </div>
                <div class="input-group">
                    <input type="text" id="user-input" class="form-control" placeholder="Type your message here...">
                    <button id="send-btn" class="btn btn-primary">Send</button>
                </div>
                <div class="loading" id="loading">Generating response, please wait...</div>
            </div>

            <script>
                function sendMessage() {
                    const userInput = document.getElementById('user-input').value;
                    if (!userInput.trim()) return;  // 如果输入为空，直接返回

                    // 禁用输入框和按钮
                    document.getElementById('user-input').disabled = true;
                    document.getElementById('send-btn').disabled = true;

                    // 显示加载提示
                    document.getElementById('loading').style.display = 'block';

                    // 添加用户消息到聊天记录
                    const chatLog = document.getElementById('chat-log');
                    chatLog.innerHTML += `
                        <div class="message user-message">
                            <div class="content">${userInput}</div>
                        </div>
                    `;

                    // 发送请求到后端
                    fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ input: userInput }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        // 添加模型回复到聊天记录
                        chatLog.innerHTML += `
                            <div class="message bot-message">
                                <div class="content">${data.response}</div>
                            </div>
                        `;
                        // 滚动到底部
                        chatLog.scrollTop = chatLog.scrollHeight;
                    })
                    .finally(() => {
                        // 启用输入框和按钮
                        document.getElementById('user-input').disabled = false;
                        document.getElementById('send-btn').disabled = false;
                        // 隐藏加载提示
                        document.getElementById('loading').style.display = 'none';
                        // 聚焦输入框
                        document.getElementById('user-input').focus();
                    });

                    // 清空输入框
                    document.getElementById('user-input').value = '';
                }

                // 绑定发送按钮点击事件
                document.getElementById('send-btn').addEventListener('click', sendMessage);

                // 绑定回车键事件
                document.getElementById('user-input').addEventListener('keypress', function (e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            </script>
        </body>
        </html>
    ''')

# 聊天接口
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    response = generate_response(user_input)
    return {'response': response}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)