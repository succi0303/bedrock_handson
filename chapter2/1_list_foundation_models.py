import boto3

bedrock = boto3.client("bedrock")

result = bedrock.list_foundation_models()

print(result)