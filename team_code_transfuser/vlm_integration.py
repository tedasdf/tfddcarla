from ollama import chat
from PIL import Image
from pydantic import BaseModel
from typing import List, Literal

class q1(BaseModel):
   Time: str
   Weather: str
   Driving_Scenario: str
   Lane_Option: str

class q2(BaseModel):
   Traffic_Lights: str
   Parked_Vehicles: str
   Building_Proximity: str

class safety_weights(BaseModel):
   w_coll: float
   w_deviation: float
   w_dis: float 
   w_speed: float

class comfort_weights(BaseModel):
   w_lat: float
   w_lon: float
   w_cent: float  


class q3(BaseModel):
   Driving_Style: Literal["Aggressive", "Conservative"]
   Level: Literal["I", "II", "III"] 
   Safety_Cost: List[safety_weights]
   Comfort_Cost: List[comfort_weights]
   Justification: str

# MODEL = 'llama3.2-vision'

class VLM():
   def __init__(self, model):
      self.query = ["Provided a detailed description of a driving scene from a set of car surround images with 6 perspectives, capturing the critical elements such as time of day, weather conditions, road environment, and available lane options.",
                    "Please list and frame the key objectives in the front view that will influence the next driving decision",
                    "Based on the previous description, should we drive conservatively or aggressively? What level and what score should we use?"
                    ]
      self.model = model

   def step(self, combined_image):

      responses = []

      # first message
      messages = [
      {'role': 'user',
         'content': self.query[0],
         'images': combined_image,
         },
      ]

      response = chat(self.model,format=q1.model_json_schema(), messages=messages)
      message = response['message']
      print(message['content'])
      messages.append(message)
      responses.append(response)
    #   print(f"Total Duration: {response.total_duration/10**9}s")
      # second message

      messages.append({'role': 'user', 'content': self.query[1]})

      response = chat(self.model,format=q2.model_json_schema(), messages=messages)
      message = response['message']
      print(message['content'])
      messages.append(message)
      responses.append(response)
    #   print(f"Total Duration: {response.total_duration/10**9}s")
      # third message
      messages.append({'role': 'user', 'content': self.query[2]})

      response = chat(self.model,format=q3.model_json_schema(), messages=messages)
      message = response['message']
      print(message['content'])
      messages.append(message)
      responses.append(response)
    #   print(f"Total Duration: {response.total_duration/10**9}s")

      return responses