from base64 import b64encode
from pydantic import BaseModel, Field
from typing import List, Literal
import requests
import json
from io import BytesIO
from traj_eval import TrajectoryScoring

URL = 'http://localhost:11434'
COMPLETIONS = '/api/chat'


class q1(BaseModel):
    Time: str
    Weather: str
    Driving_Scenario: str
    Lane_Option: str


class q2(BaseModel):
    Traffic_Lights: str
    Parked_Vehicles: str
    Building_Proximity: str


class q3(BaseModel):
    Driving_Style: Literal["Aggressive", "Conservative"]
    Level: Literal["I", "II", "III"]
    Weight_Collision: float = Field(..., gt=0, lt=10)
    Weight_Deviation: float = Field(..., gt=0, lt=10)
    Weight_Distance: float = Field(..., gt=0, lt=10)
    Weight_Speed: float = Field(..., gt=0, lt=10)
    Weight_Lat: float = Field(..., gt=0, lt=10)
    Weight_Lon: float = Field(..., gt=0, lt=10)
    Weight_Cent: float = Field(..., gt=0, lt=10)
    Justification: str

# MODEL = 'llama3.2-vision'


class VLM():
    def __init__(self, model="llama3.2-vision"):
        self.query = ["Provided a detailed description of a driving scene from a set of car surround images with 6 perspectives, capturing the critical elements such as time of day, weather conditions, road environment, and available lane options.",
                      "Please list and frame the key objectives in the front view that will influence the next driving decision",
                      "Based on the previous description, should we drive conservatively or aggressively? What level and what score should we use?"
                      ]
        self.model = model

    def chat_model(self, format, messages):
        payload = {"model": self.model,
                   "messages": messages,
                   "format": format,
                   "stream": False
                   }
        response = requests.post(URL + COMPLETIONS, json=payload)
        if response.status_code != 200:
            raise Exception("Error: Server responded with ",
                            response.status_code)

        return response.json()

    def step(self, combined_image, weights):

        responses = []

        buffered = BytesIO()
        combined_image.save(buffered, format="JPEG")
        encoded_image = b64encode(buffered.getvalue()).decode("utf-8")
        # first message
        messages = [
            {
                'role': 'system',
                'content': f"""
                You are a bot for defining the weights of the following metrics:

                Safety Metrics:
                Weight_Collision: a function that increases collision penalty as the vehicle gets closer to an obstacle, so near obstacles have much higher risk than far ones. Initial: {weights.w_coll:.2f}  
                Weight_Deviation: a penalty that increases as the vehicle moves further from the desired lane or path. Initial: {weights.w_dev:.2f}  
                Weight_Distance: a penalty that increases when the vehicle’s distance to the goal becomes longer than necessary. Initial: {weights.w_dis:.2f}  
                Weight_Speed: a penalty for speeds that are too high or too low compared to the desired speed profile. Initial: {weights.w_speed:.2f}  

                Comfort Metrics:
                Weight_Lat: a penalty for high sideways (lateral) acceleration that could cause discomfort. Initial: {weights.w_lat:.2f}  
                Weight_Lon: a penalty for high forward/backward (longitudinal) acceleration changes that could cause discomfort. Initial: {weights.w_lon:.2f}  
                Weight_Cent: a penalty for high centripetal acceleration when turning, linked to cornering comfort. Initial: {weights.w_cent:.2f}  
                """
            },
            {'role': 'user',
                'content': self.query[0],
             'images': [encoded_image],
             },
        ]

        response = self.chat_model(
            format=q1.model_json_schema(), messages=messages)
        message = response['message']
        print(message['content'])
        messages.append(message)
        responses.append(response)
        #   print(f"Total Duration: {response.total_duration/10**9}s")
        # second message

        messages.append({'role': 'user', 'content': self.query[1]})

        response = self.chat_model(
            format=q2.model_json_schema(), messages=messages)
        message = response['message']
        print(message['content'])
        messages.append(message)
        responses.append(response)
        #   print(f"Total Duration: {response.total_duration/10**9}s")
        # third message
        messages.append({'role': 'user', 'content': self.query[2]})

        response = self.chat_model(
            format=q3.model_json_schema(), messages=messages)
        message = response['message']
        print(message['content'])
        messages.append(message)
        responses.append(response)
        #   print(f"Total Duration: {response.total_duration/10**9}s")

        return responses
