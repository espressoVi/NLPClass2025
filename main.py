from Generator import LLM

def main():
    system_message = "You are a helpful AI assistant."\
                     "Please assist the user with their query."
    user_message   = "Tell me a joke about a chicken."

    prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
    ]

    llm = LLM()
    output = llm(prompt)
    print()
    print(output[0])
    print()


if __name__ == "__main__":
    main()
