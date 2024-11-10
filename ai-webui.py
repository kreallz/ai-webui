import time
import gradio as gr
from openai import OpenAI

theme = gr.themes.Default().set(
    body_text_color="*neutral_100",
    color_accent_soft="*neutral_700",
    background_fill_primary="*neutral_950",
    background_fill_secondary="*neutral_900",
    border_color_accent="*neutral_600",
    border_color_primary="*neutral_700",
    link_text_color_active="*secondary_500",
    link_text_color="*secondary_500",
    link_text_color_hover="*secondary_400",
    link_text_color_visited="*secondary_600",
    body_text_color_subdued="*neutral_400",
    block_background_fill="*neutral_800",
    block_label_text_color="*neutral_200",
    block_title_text_color="*neutral_200",
    code_background_fill="*neutral_800",
    checkbox_background_color="*neutral_800",
    checkbox_border_color="*neutral_700",
    checkbox_border_color_hover="*neutral_600",
    checkbox_label_background_fill="*neutral_800",
    error_background_fill="*neutral_900",
    input_background_fill="*neutral_800",
    input_border_color_focus="*neutral_700",
    input_placeholder_color="*neutral_500",
    table_border_color="*neutral_700",
    table_even_background_fill="*neutral_950",
    table_odd_background_fill="*neutral_900",
    button_primary_background_fill_hover="*primary_700",
    button_primary_border_color="*primary_600",
    button_primary_border_color_hover="*primary_500",
    button_secondary_background_fill="*neutral_600",
    button_secondary_background_fill_hover="*neutral_700",
    button_secondary_border_color="*neutral_600",
    button_secondary_border_color_hover="*neutral_500",
    color_accent="*primary_500",
)

#config
model_name = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
prompt = "You are a helpful personal assistant that will help the user with anything they ask"
api_url = "http://localhost:1234/v1"
api_key = "lm-studio"

client = OpenAI(base_url = api_url, api_key = api_key)

def slow_echo(message, conversation_history):
    if not conversation_history:
        conversation_history.append({"role": "system", "content": prompt})
    
    conversation_history.append({"role": "user", "content": message})
    
    streamed_completion = client.chat.completions.create(
        model=model_name,
        messages=conversation_history,
        stream=True
    )
    
    try:
        content = ""
        history_record = {"role": "system", "content": content}
        conversation_history.append(history_record)
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                content += delta_content
                yield content
    finally:
      streamed_completion.response.close()

with gr.ChatInterface(slow_echo, type="messages", theme = theme, fill_width = True) as demo:
    demo.chatbot.label = model_name

# TODO custom buttons for start/abort streamed_completion

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7777, show_error=True)