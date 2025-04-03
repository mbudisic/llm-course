# You can find this code for Chainlit python streaming here
# (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import openai
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import (
    ChatOpenAI,
)  # importing ChatOpenAI tools
from dotenv import load_dotenv
import os

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a helpful assistant who always speaks in a pleasant tone!

"""

user_template = """{input}
Think through your response step by step unless asked to be concise. In that
case, use no more than 2 sentences.
"""


vibe_check = {
    "intro": "Please introduce yourself",
    "explain": "Explain the concept of object-oriented programming in simple terms to a complete beginner.",
    "summarize": """
    Read the following paragraph and provide a concise summary of the key points:
    
    It is clear enough that some of Kierkegaard's early writings contained disguised 
    communications to Regine, who soon married another man (Johan Frederik Schlegel, 
    later Governor-General of the Danish West Indies), which meant that Kierkegaard 
    could not communicate directly with her. He hoped she might come to realize that 
    he still loved her, but also understand why he could not go through with the 
    marriage. Biographies of Kierkegaard often focus on the broken engagement in 
    detail, and commentators have sometimes gone to great 
    lengths to find "messages to Regine" in Kierkegaard's texts, some more plausible than others. However, 
    Regine was far from the only person who could be described as "that individual" to whom Kierkegaard 
    dedicated many of his works.
    """,
    "create": "Write a short, imaginative story (100–150 words) about a robot finding "
    "friendship in an unexpected place.",
    "math": "If a store sells apples in packs of 4 and oranges in packs of 3, how many "
    "packs of each do I need to buy to get exactly 12 apples and 9 oranges?",
    "formalize": """
    Rewrite the following paragraph in a professional, formal tone:

    My Adidas walk through concert doors
    And roam all over coliseum floors
    I stepped on stage, at Live Aid
    All the people gave, and the poor got paid
    And out of speakers I did speak
    I wore my sneakers but I'm not a sneak
    My Adidas touch the sand of a foreign land
    With mic in hand, I cold took command
    My Adidas and me close as can be
    We make a mean team, my Adidas and me
    We get around together, we're down forever
    And we won't be mad when caught in bad weather
    """,
    "code_edit": """
    Convert the following recursion written in C into a flat for loop in Python:
    
    #include<stdio.h>
    long int multiplyNumbers(int n);
    int main() {
        int n;
        printf("Enter a positive integer: ");
        scanf("%d",&n);
        printf("Factorial of %d = %ld", n, multiplyNumbers(n));
        return 0;
    }
    long int multiplyNumbers(int n) {
        if (n>=1)
            return n*multiplyNumbers(n-1);
        else
            return 1;
    }
    """,
}


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    # Create settings with default model
    my_model = os.getenv("BEYOND_MODEL", "gpt-3.5-turbo")

    model_list = [model.id for model in openai.models.list()]
    print(model_list)

    if my_model not in model_list:
        print(f"Model {my_model} not found. Using default model gpt-3.5-turbo.")
        my_model = "gpt-3.5-turbo"

    settings = {
        "model": my_model,
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)

    # add button actions for every vibe check value
    actions = [
        cl.Action(
            name=f"vibe_{vibe_key}_button",
            icon="mouse-pointer-click",
            value=vibe_value,
            label=vibe_key,
            type="run",
        )
        for (vibe_key, vibe_value) in vibe_check.items()
    ]

    # create assistant message with buttons and model field
    msg = cl.Message(
        actions=actions,
        author="assistant",
        content="Care for some vibe checks?",
    )

    await cl.Message(
        author="assistant",
        content=f"""
                     Running on {my_model}.
                     Other options are {model_list}""",
    ).send()
    await msg.send()


async def on_action(action: cl.Action):
    msg = cl.Message(content=action.value, author="user")
    await msg.send()  # display the message on screen as if typed by a user
    await send_message(msg)  # send the message to LLM for response


for vibe_key in vibe_check.keys():
    on_action = cl.action_callback(f"vibe_{vibe_key}_button")(on_action)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    await send_message(message)


async def send_message(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = openai.AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
