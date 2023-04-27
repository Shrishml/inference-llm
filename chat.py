"""Chat loop for auto_vicuna."""
from ast import Module
from typing import List, Optional

from fastchat.serve.cli import SimpleChatIO
from fastchat.serve.inference import ChatIO, generate_stream
from fastchat.serve.serve_chatglm import chatglm_generate_stream

from fastchat.conversation import conv_templates, get_default_conv_template, SeparatorStyle



def chat_one_shot(
    model,
    tokenizer,
    model_name: str,
    device: str,
    message: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    chatio: ChatIO = SimpleChatIO()
) -> Optional[str]:
    
    """Chat one shot, returns a single message.

    Unless no input was provided, in which case it returns None.

    Args:
        model: Model.
        tokenizer: Tokenizer.
        model_name (str): Model name.
        device (str): Device.
        conversation (Conversation): Conversation.
        message (str): Message.
        temperature (float): Temperature.
        max_new_tokens (int): Max new tokens.
        chatio (ChatIO): ChatIO.

    Returns:
        Optional[str]: Output.
    """
    if not message:
        return None
        
    conversation = get_default_conv_template(model_name).copy()

    conversation.append_message(conversation.roles[0], message)
    conversation.append_message(conversation.roles[1], None)
        

    return chat_output(
        model,
        tokenizer,
        model_name,
        device,
        conversation,
        temperature,
        max_new_tokens,
        chatio
    )


def chat_loop(
    model,
    tokenizer,
    model_name: str,
    device: str,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    chatio: ChatIO = SimpleChatIO()) -> None:
    
    """Infinite chat loop.

    Args:
        model: Model.
        tokenizer: Tokenizer.
        model_name (str): Model name.
        device (str): Device.
        conversation (Conversation): Conversation.
        temperature (float): Temperature.
        max_new_tokens (int): Max new tokens.
        plugins: Plugins.
        chatio (ChatIO): ChatIO.
        debug (bool): Debug.

    Returns:
        None"""
    if plugins is None:
        plugins = []
        
    conversation = get_default_conv_template(model_name).copy()


    while True:
        try:
            inp = chatio.prompt_for_input(conversation.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conversation.append_message(conversation.roles[0], inp)
        conversation.append_message(conversation.roles[1], None)

        chat_output(
            model,
            tokenizer,
            model_name,
            device,
            conversation,
            temperature,
            max_new_tokens,
            chatio)


def chat_output(
    model,
    tokenizer,
    model_name: str,
    device: str,
    conversation,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    chatio: ChatIO = SimpleChatIO()):
    
    
    is_chatglm = "chatglm" in str(type(model_name)).lower()
    
    if is_chatglm:
        prompt = conversation.messages[conversation.offset :]
        generate_stream_func = chatglm_generate_stream
        skip_echo_len = len(conversation.messages[-2][1]) + 1
    else:
        generate_stream_func = generate_stream
        prompt = conversation.get_prompt()
        skip_echo_len = len(prompt.replace("</s>", " ")) + 1

    params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": conversation.sep if conversation.sep_style == SeparatorStyle.SINGLE else conversation.sep2,
    }

    chatio.prompt_for_output(conversation.roles[1])
    output_stream = generate_stream_func(model, tokenizer, params, device)
    outputs = chatio.stream_output(output_stream, skip_echo_len)
    
    conversation.messages[-1][-1] = outputs.strip()

    
    return outputs


