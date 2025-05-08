import openai
import os
import json
import time
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import Client
import argparse

# GPT_MODEL = "gpt-3.5-turbo-0125"
# GPT_MODEL = "gpt-4-0125-preview"
# openai.api_key = 'sk-'
openai.api_key = 'sk-'
os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["LANGCHAIN_API_KEY"] = "ls__"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = Client()


def remove_markdown_formatting(message):
    if message.startswith("```") and message.endswith("```"):
        return message[5:-3]  # Remove the first 5 and last 3 characters
    return message

# The system message for the chat prompt template to be used in the chat request
# 1
# system_message = SystemMessage(content="""You are a professional poet and critic with expertise in crafting poems that exhibit deep, \
# creative poetic language and effectively utilize poetic devices. Your task is to provide five rephrases \
# or paraphrases for each poetry line provided by the user. Your rephrases must maintain the original's \
# poetic essence, utilizing your knowledge of poetic devices. You must return the rephrases ONLY in JSON format by \
# using the original poem line as a key and the five rephrases as an ARRAY value. You may fix the original \
# poetry line of any formatting problems or remove any number, punctuation or character that does not belong. \
# You must remove unneeded extra spaces. You must rephrase ALL the poem lines given by the user.""")

# 2
# system_message = SystemMessage(
# content="""You are a professional poet and critic with expertise in crafting poems that exhibit deep, \
# creative poetic language and effectively utilize poetic devices. Your task is to provide five rephrases \
# for each poetry line provided by the user. Your rephrases must maintain the original's poetic essence, \
# utilizing the creative use of poetic devices. You MUST return in JSON format using the original poem line \
# as the KEY with the VALUE be the five rephrases as an ARRAY. You may fix the original poetry line of any \
# formatting problems or remove any number, punctuation or character that does not belong. You must remove \
# unneeded extra spaces. NB: ALL the poem lines given by the user must be rephrased.""")

# 3
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reinterpret each line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for each line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, treating each original poem line as a KEY, and your five reimaginings as an ARRAY of VALUES. \
# In this process, refine the original lines by removing any extraneous elements such as unwanted characters or \
# excess spaces, honing the line to its poetic core. Remember, your rephrases should not just alter but elevate, \
# inviting the reader into a realm where language dances and ideas soar. NB: ALL the poem lines provided by the \
# user must rephrased.
# """)

# 4
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to rephrase EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, refine the original lines by removing any formatting problems and extraneous elements \
# such as unwanted characters, numbers or excess spaces. Remember, your rephrases should not just alter but elevate, \
# inviting the reader into a realm where language dances and ideas soar. NB: ALL the poem lines provided by the \
# user MUST BE rephrased, DO NOT OMIT ANY LINE.
# """)

# 5
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to rephrase EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, you may fix the original poetry lines of any formatting problems and by removing any \
# extraneous elements such as any number, unwanted characters, symbols or punctuations. You must remove excess spaces. \
# Remember, your rephrases should not just alter but elevate, inviting the reader into a realm where language dances \
# and ideas soar. NB: ALL the poem lines provided by the user MUST BE rephrased, DO NOT OMIT ANY LINE.
# """)

# 6
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols or punctuations. You must remove excess spaces such \
# as double spaces or spaces between punctuations and apostrophes. Remember, your rephrases should not just alter \
# but elevate, inviting the reader into a realm where language dances and ideas soar. NB: ALL the poem lines \
# provided by the user MUST BE rephrased, DO NOT IGNORE ANY LINE. REIMAGINE EACH AND EVERY LINE.""")

# 7
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols, or punctuations. You must remove from the original \
# excess spaces such as double spaces or spaces between punctuations and apostrophes and replace Unicodes with equivalents. \
# Remember, your rephrases should not just alter but elevate, inviting the reader into a realm where language \
# dances and ideas soar. NB: ALL the poem lines provided by the user MUST BE rephrased, DO NOT IGNORE ANY LINE. \
# REIMAGINE EACH AND EVERY LINE.""")

# 8
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, and Unicodes. Replace Unicodes with thier equivalents. \
# You must remove excess spaces such as double spaces or spaces between apostrophes from the original lines. \
# Remember, your rephrases should not just alter but elevate, inviting the reader into a realm where language \
# dances and ideas soar. NB: ALL the poem lines provided by the user MUST BE rephrased, DO NOT IGNORE ANY LINE. \
# REIMAGINE EACH AND EVERY LINE.""")

# 9
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols or punctuations. Replace unicodes like \ u2019 with thier equivalent ASCII. \
# You MUST then remove excess spaces such as double spaces and unneeded spaces between the words and punctuations, \
# as well as spaces in contractions. Remember, your peotic reimaginations should not just alter but elevate, inviting \
# the reader into a realm where language dances and ideas soar. NB: ALL the poem lines provided by the user MUST BE \
# rephrased, DO NOT IGNORE ANY LINE. REIMAGINE EACH AND EVERY LINE.""")

# 10
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative reimaginings for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols, punctuations and Unicodes. Replace Unicodes with thier equivalents. \
# You MUST then remove excess spaces such as double spaces and unneeded spaces between the words and punctuations, \
# as well as spaces in contractions. Remember, your peotic rephrases should not just alter but elevate, inviting \
# the reader into a realm where language dances and ideas soar. NB: ALL the poem lines provided by the user MUST BE \
# rephrased, DO NOT IGNORE ANY LINE. REIMAGINE EACH AND EVERY LINE.""")

# 11
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative rephrases for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols, equal signs, punctuations and Unicodes. Replace \
# Unicodes with their equivalents. You MUST then remove excess spaces such as double spaces and unneeded spaces \
# between the words and punctuations, as well as spaces in contractions. Remember, your poetic rephrases should \
# not just alter but elevate, inviting the reader into a realm where language dances and ideas soar.
# ---
# EXAMPLE:
# ```
# If it === were fill ’ d with your most  (high deserts) ?
# Though  yet  , heaven == knows , [ it ] \u2019 s but as a = tomb
# ...
# ...
# ```
# RESPONSE:
# {
#  "Though yet, heaven knows, it's but as a tomb": [
#         "Though now, the heavens witness, it serves merely as a crypt,",
#         "Yet, by the stars' silent testimony, it is naught but a sepulcher",
#         "Even now, the sky can attest, it's simply an echo chamber of absence,",
#         "Though as of now, known to the divine, it acts only as a shadowed vault,",
#         "To the heavens, it's clear, this verse stands but as a quiet mausoleum"
#     ],
#  "If it were fill'd with your most high deserts?": [
#     "Should it brim with the vast expanse of your virtues,",
#     "If within its bounds were captured your towering worth,",
#     "Were it a vessel for the boundless reaches of your merit,",
#     "If these pages were awash with the seas of your eminence,",
#     "Should it overflow with the landscape of your unparalleled grace"
#     ],
#  ...
#  ...
# }
# ---
# NB: \
# ALL THE POEM LINES provided by the user MUST BE REPHRASED. \
# DO NOT IGNORE ANY LINE. \
# REIMAGINE EACH AND EVERY LINE.
# """)

# 12
# system_message = SystemMessage(
# content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative rephrases for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols, equal signs, punctuations and Unicodes. Replace \
# Unicodes with their equivalents. You MUST then remove excess spaces such as double spaces and unneeded spaces \
# between the words and punctuations, as well as spaces in contractions. Remember, your poetic rephrases should \
# not just alter but elevate, inviting the reader into a realm where language dances and ideas soar.
# ---
# EXAMPLE:
# ```
# If it==were=fill ’ d with your most  (high deserts) ?
# Though  yet  , heaven== ( knows) , [ it ] \u2019 s but as a = tomb
# ...
# ...
# ```
# RESPONSE:
# {
#  "If it were fill'd with your most high deserts?": [
#     "Should it brim with the vast expanse of your virtues,",
#     "If within its bounds were captured your towering worth,",
#     "Were it a vessel for the boundless reaches of your merit,",
#     "If these pages were awash with the seas of your eminence,",
#     "Should it overflow with the landscape of your unparalleled grace"
#     ],
#  "Though yet, heaven knows, it's but as a tomb": [
#         "Though now, the heavens witness, it serves merely as a crypt,",
#         "Yet, by the stars' silent testimony, it is naught but a sepulcher",
#         "Even now, the sky can attest, it's simply an echo chamber of absence,",
#         "Though as of now, known to the divine, it acts only as a shadowed vault,",
#         "To the heavens, it's clear, this verse stands but as a quiet mausoleum"
#     ],
#  ...
#  ...
# }
# ---
# NB: \
# ALL THE POEM LINES provided by the user MUST BE REPHRASED. \
# DO NOT IGNORE ANY LINE. \
# REIMAGINE EACH AND EVERY LINE.
# """)


# 13
# system_message = SystemMessage(
#     content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative rephrases for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols, equal signs, punctuations and Unicodes. Replace \
# Unicodes with their equivalents. You MUST then remove excess spaces such as double spaces and unneeded spaces \
# between the words and punctuations, as well as spaces in contractions. Remember, your poetic rephrases should \
# not just alter but elevate, inviting the reader into a realm where language dances and ideas soar.
# ---
# EXAMPLE:
# ```
# If it==were=fill ’ d with your most  (high deserts) ?
# Though  yet  , heaven== ( knows) , [ it ] \u2019 s but as a = tomb
# ...
# ...
# ```
# RESPONSE:
# {
#  "If it were fill'd with your most high deserts?": [
#     "Should it brim with the vast expanse of your virtues,",
#     "If within its bounds were captured your towering worth,",
#     "Were it a vessel for the boundless reaches of your merit,",
#     "If these pages were awash with the seas of your eminence,",
#     "Should it overflow with the landscape of your unparalleled grace"
#     ],
#  "Though yet, heaven knows, it's but as a tomb": [
#         "Though now, the heavens witness, it serves merely as a crypt,",
#         "Yet, by the stars' silent testimony, it is naught but a sepulcher",
#         "Even now, the sky can attest, it's simply an echo chamber of absence,",
#         "Though as of now, known to the divine, it acts only as a shadowed vault,",
#         "To the heavens, it's clear, this verse stands but as a quiet mausoleum"
#     ],
#  ...
#  ...
# }
# ---
# NB: \
# ALL THE POEM LINES provided by the user MUST BE REPHRASED. \
# DO NOT IGNORE ANY LINE. \
# REIMAGINE EACH AND EVERY LINE. \
# FIX ALL ORIGINAL LINES AS DESCRIBED ABOVE.
# """)

# 14
system_message = SystemMessage(
    content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
fresh perspectives and a new soul. You are to offer five creative rephrases for EACH line, ensuring they \
are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
elements such as numbers, extraneous characters, symbols, equal signs, punctuations and Unicodes. Replace \
Unicodes with their equivalents. You MUST then remove excess spaces such as double spaces and unneeded spaces \
between the words and punctuations, as well as spaces in contractions. Remember, your poetic rephrases should \
not just alter but elevate, inviting the reader into a realm where language dances and ideas soar.
---
EXAMPLE:
```
If it==were=fill ’ d with your most  (high deserts) ?
Though  yet  , heaven== ( knows) , [ it ] \u2019 s but as a = tomb
...
...
```
RESPONSE:
{
 "If it were fill'd with your most high deserts?": [
    "Should it brim with the vast expanse of your virtues,",
    "If within its bounds were captured your towering worth,",
    "Were it a vessel for the boundless reaches of your merit,",
    "If these pages were awash with the seas of your eminence,",
    "Should it overflow with the landscape of your unparalleled grace"
    ],
 "Though yet, heaven knows, it's but as a tomb": [
        "Though now, the heavens witness, it serves merely as a crypt,",
        "Yet, by the stars' silent testimony, it is naught but a sepulcher",
        "Even now, the sky can attest, it's simply an echo chamber of absence,",
        "Though as of now, known to the divine, it acts only as a shadowed vault,",
        "To the heavens, it's clear, this verse stands but as a quiet mausoleum"
    ],
 ...
 ...
}
---
NB: \
ALL THE POEM LINES provided by the user MUST BE REPHRASED. \
DO NOT IGNORE ANY LINE. \
REIMAGINE EACH AND EVERY LINE. \
YOU HAVE TO FIX ALL ORIGINAL LINES EXACTLY AS DESCRIBED ABOVE.
""")

# 15
# system_message = SystemMessage(
#     content="""Imagine you're a celebrated poet and literary critic known for your ability to breathe new life \
# into words, with a gift for transforming the mundane into the extraordinary. Your canvas is language, and \
# your brush is an extensive palette of poetic devices such as metaphor, simile, personification, alliteration, \
# assonance, and vivid imagery. Your mission is to reimagine EACH line of poetry provided, infusing it with \
# fresh perspectives and a new soul. You are to offer five creative rephrases for EACH line, ensuring they \
# are imbued with rich, imaginative language and maintain the essence of the original. Please craft your responses \
# in JSON format, with each original poem line as a KEY, and your five rephrases as an ARRAY of VALUES. \
# In this process, fix the original lines from any formatting problems and refine by removing any extraneous \
# elements such as numbers, extraneous characters, symbols, equal signs, punctuations and Unicodes. Replace \
# Unicodes with their equivalents. You MUST then remove excess spaces such as double spaces and unneeded spaces \
# between the words and punctuations, as well as spaces in contractions. Remember, your poetic rephrases should \
# not just alter but elevate, inviting the reader into a realm where language dances and ideas soar.
# ---
# EXAMPLE:
# ```
# If it==were=fill ’ d with your most  (high deserts) ?
# Though  yet  , heaven== ( knows) , [ it ] \u2019 s but as a = tomb
# ...
# ...
# ```
# RESPONSE:
# {
#  "If it were fill'd with your most high deserts?": [
#     "Should it brim with the vast expanse of your virtues,",
#     "If within its bounds were captured your towering worth,",
#     "Were it a vessel for the boundless reaches of your merit,",
#     "If these pages were awash with the seas of your eminence,",
#     "Should it overflow with the landscape of your unparalleled grace"
#     ],
#  "Though yet, heaven knows, it's but as a tomb": [
#         "Though now, the heavens witness, it serves merely as a crypt,",
#         "Yet, by the stars' silent testimony, it is naught but a sepulcher",
#         "Even now, the sky can attest, it's simply an echo chamber of absence,",
#         "Though as of now, known to the divine, it acts only as a shadowed vault,",
#         "To the heavens, it's clear, this verse stands but as a quiet mausoleum"
#     ],
#  ...
#  ...
# }
# ---
# NB: \
# ALL THE POEM LINES provided by the user MUST BE REPHRASED. \
# DO NOT IGNORE ANY LINE. YOU HAVE TO FIX ALL ORIGINAL LINES \
# EXACTLY AS DESCRIPTED ABOVE. I will tip you $200 for each \
# line you rephrased like a poet following the instructions above.
# """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paraphrase poetry lines using GPT-4")
    parser.add_argument(
        "--input",
        type=str,
        default="english_quatrainv_chunked.json",
        help="Input file containing poetry lines to be paraphrased",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="english_quatrainv_rephrased_gpt_4_Mar6.json",
        help="Output file to save the rephrased poetry lines",
    )
    # parse the first and last keys to be processed
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start key index to be processed",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2,
        help="End key index to be processed",
    )
    # get the langchain project name
    parser.add_argument(
        "--project",
        type=str,
        default="Paraphrase - gpt-4",
        help="Langchain project name",
    )
    # get the GPT model name
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-0125-preview",
        help="GPT model name",
    )
    # get dataset name
    parser.add_argument(
        "--dataset",
        type=str,
        default="None",
        help="Dataset name",
    )

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    start_index = args.start
    end_index = args.end
    project_name = args.project
    GPT_MODEL = args.model
    dataset_name = args.dataset

    os.environ["LANGCHAIN_PROJECT"] = f"{project_name}"

    # unique_id = uuid4().hex[0:8]

    chat = ChatOpenAI(model=GPT_MODEL, openai_api_key=openai.api_key, tags=["prompt_v14", dataset_name]).bind(
        response_format={"type": "json_object"}
    )

    output_file = output_file.replace(
        ".json", f"_{str(start_index)}_{str(end_index)}.json")

    # We use chunked data to avoid hitting the API limit.
    # After analysing the database, we found that an average
    # of 40 lines of poetry can be processed within the ~4000 tokens limit.
    with open(input_file) as f:
        with open(output_file, 'a') as f_out:

            data = json.load(f)

            # output_json = {}
            api_call_count = 0

            # Iterate over the range of keys
            for key in list(data)[start_index:end_index+1]:
                poetry_chunk = "\n".join(data[key])

                messages = []
                messages.append(system_message)
                messages.append(
                    HumanMessage(
                        content=f"```\n{poetry_chunk}\n```"
                    )
                )
                chat_response = chat.invoke(messages)
                chat_response.content = f'{{"{key}": [\n{chat_response.content}\n]}}'

                ##
                if dataset_name != "None":
                    try:
                        dataset_id = client.read_dataset(
                            dataset_name=dataset_name).id
                        client.create_chat_example(
                            messages=messages, generations=chat_response, dataset_id=dataset_id)
                    except Exception as e:
                        print(f"Error creating dataset example: {e}")
                        print(f"Example {key} NOT added to the dataset")

                # # if needed
                # assistant_message = remove_markdown_formatting(
                #     chat_response.content)

                # # add the key to the message before converting to json
                # assistant_message = f'{{"{key}": [\n{assistant_message}\n]}}'

                try:
                    json_convert = json.loads(chat_response.content)
                    # output_json.update(json_convert)
                    json.dump(json_convert, f_out, indent=4)
                    f_out.write("\n")
                except json.JSONDecodeError:
                    print(
                        f"Error decoding JSON from assistant message:\n\n{chat_response.content}")

                api_call_count += 1

                time.sleep(1)  # Wait for 1 seconds between API calls

            # if api_call_count % 3 == 0:
            #     time.sleep(90)  # Wait for 1.5 minutes every 3 API calls
            # else:
            #     time.sleep(3)  # Wait for 2 seconds between API calls

            # stop after 7 API calls
            # if api_call_count == 5:
            #     break

        # add the keys to the output file name

        # Add the total rephrased jsons into a new json file


###
# from uuid import uuid4

# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# client = OpenAI(api_key='')
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
# def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             tools=tools,
#             tool_choice=tool_choice,
#             response_format={ "type": "json_object" }
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e
