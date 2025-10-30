#!/bin/python3.13
from openai import OpenAI, OpenAIError
import re
import time
import os


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],       # Put your API key here.
)

def _call_with_retry(fn, max_retries=5, base_delay=1, **kwargs):
    for attempt in range(max_retries):
        try:
            resp = fn(**kwargs)
            return resp.choices[0].message.content
        except (OpenAIError, AttributeError) as exc:
            if attempt == max_retries - 1: raise
            sleep_for = base_delay * (2 ** attempt)
            time.sleep(sleep_for)

def get_response(prompt):
    response = _call_with_retry(
        client.chat.completions.create,
        model      = "openai/gpt-oss-20b:free",     # Replace with (free) model of your choice.
        messages   = prompt,
        max_tokens = 2048,
    )
    return response


def main():
    """
        The following is how a prompt will usually be structured. The system
        prompt should contain high level instructions, in-context demonstrations,
        list of capabilities, rules to follow, etc.

        The user prompt is the query given by the user: In almost all cases this
        will be given by the end user.
    """
    prompt = [
        {"role": "system", "content": "You are a helpful AI assistant. Assist the user with their query."},
        {"role": "user", "content": "Write a haiku about a river."},
    ]
    response = get_response(prompt)
    print("First response:\n", response)

    # For multi-turn conversations prompts can be appended to the prompt array.
    prompt.append({"role": "assistant", "content": response})
    prompt.append({"role": "user", "content": "Now write a short poem."})

    # Now let's get the second response.
    new_response = get_response(prompt)
    print("\nSecond response:\n", new_response)

if __name__ == "__main__":
    main()

