import gradio as gr

# 定义一个函数
def greet(name):
    return f"Hello {name}!"

# 用 gradio 接口包装
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# 启动本地 Web 界面
demo.launch()
