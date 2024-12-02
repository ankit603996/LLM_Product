import os

os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ankit-m469twgs-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"
os.environ["AZURE_OPENAI_API_KEY"] = "2QwGgJ4m2sxUq2AQZWyYTv5cvRKPh2WSUfuGnYda9AB6Fswz0guGJQQJ99ALACHYHv6XJ3w3AAAAACOGzKKj"

# Import Azure OpenAI
from langchain_openai import AzureOpenAI
# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name="gpt-4-turbo-2024-04-09",
)
# Run the LLM
llm.invoke("Tell me a joke")

# Model version
# turbo-2024-04-09
# Deployment name
# gpt-4
# Model name
# gpt-4
# gpt-4-turbo-2024-04-09: This is the GPT-4 Turbo with Vision GA model. The context window is 128,000 tokens, and it can return up to 4,096 output tokens. The training data is current up to December 2023.
#
# gpt-4-1106-preview (GPT-4 Turbo): The latest gpt-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. It returns a maximum of 4,096 output tokens. This preview model is not yet suited for production traffic. Context window: 128,000 tokens. Training Data: Up to April 2023.
#
# gpt-4-vision Preview (GPT-4 Turbo with vision): This multimodal AI model enables users to direct the model to analyze image inputs they provide, along with all the other capabilities of GPT-4 Turbo. It can return up to 4,096 output tokens. As a preview model version, it is not yet suitable for production traffic. The context window is 128,000 tokens. Training data is current up to April 2023.
#
# gpt-4-0613: gpt-4 model with a context window of 8,192 tokens. Training data up to September 2021.
#
# gpt-4-0314: gpt-4 legacy model with a context window of 8,192 tokens. Training data up to September 2021. This model version will be retired no earlier than July 5, 2024.


import openai

# Set up Azure OpenAI credentials
openai.api_type = "azure"
openai.api_base = "https://ankit-m469twgs-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"  # Replace with your Azure endpoint
openai.api_version = "2023-05-15"  # Update to the supported API version
openai.api_key = "2QwGgJ4m2sxUq2AQZWyYTv5cvRKPh2WSUfuGnYda9AB6Fswz0guGJQQJ99ALACHYHv6XJ3w3AAAAACOGzKKj"  # Replace with your Azure API key
# Perform inference using GPT-4
system_message =  "You are a helpful assistant."
user_message = "What are the talent products delivered by DASA?"
response = openai.ChatCompletion.create(
    engine="gpt-4",  # Replace with your deployment name in Azure
    messages=[
        {"role": "system", "content":system_message},
        {"role": "user", "content": user_message}
    ],
    max_tokens=100,
    temperature=0.7  # Adjust for creativity
)
# Print the response
print(response['choices'][0]['message']['content'])
