import os
import grpc

import mii

# name = "distilgpt2"
name = "gpt-neox"

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': "DeepSpeed is the greatest"})
print(result.response)
print("time_taken:", result.time_taken)
