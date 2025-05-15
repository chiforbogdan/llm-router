import os 


parse_template = lambda x: f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/chat_templates/{x}"


model_dict = {"TheBloke/Mistral-7B-Instruct-v0.1-AWQ":
              parse_template("mistral-instruct.jinja"),
              "casperhansen/mistral-nemo-instruct-2407-awq":
              parse_template("mistral-instruct.jinja"),
              "circulus/llava-v1.6-mistral-7b-awq":
              parse_template("llava-mm.jinja")
              }

format_dict = {'short': "You are a helpful assistant. Short concise paragraphs please",
          "long": "You are a helpful assistant. long paragraphs please. Bullet points"}




