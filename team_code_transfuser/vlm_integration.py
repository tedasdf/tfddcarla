from base64 import b64encode
from pydantic import BaseModel, Field
from typing_extensions import Literal
import requests
import json
from io import BytesIO
# from traj_eval import TrajectoryScoring

URL = 'http://localhost:11434'
COMPLETIONS = '/api/chat'


class q1(BaseModel):
    Time: str = Field(..., description="Specify time of day in detail, e.g. '10:00 AM, mid-morning rush hour'")
    Weather: str = Field(..., description="Describe the weather in full, e.g. 'Sunny with light clouds'")
    Driving_Scenario: str = Field(..., description="Describe the road type and conditions, e.g. 'Highway with moderate traffic'")
    Lane_Option: str = Field(..., description="Describe which lane is chosen and why, e.g. 'Left lane for overtaking slower cars'")

class q2(BaseModel):
    Traffic_Lights: str = Field(..., description="Traffic light state, e.g. 'Red light with countdown timer visible'")
    Parked_Vehicles: str = Field(..., description="Describe number and placement, e.g. '3 cars parked closely along right side'")
    Building_Proximity: str = Field(..., description="Describe closeness of buildings, e.g. 'Shops immediately adjacent to roadside'")

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
    def __init__(self, model="llama3.2-vision:11b", memory=128):
        self.query = ["Provided a detailed description of a driving scene from a set of car surround images with 6 perspectives, capturing the critical elements such as time of day, weather conditions, road environment, and available lane options.",
                      "Please list and frame the key objectives in the front view that will influence the next driving decision",
                      "Based on the previous description, should we drive conservatively or aggressively? What level and what score should we use?"
                      ]
        self.model = model
        self.messages = []
        self.memory = memory

    def chat_model(self,format, messages):

        payload = {"model": self.model,
                   "messages": messages,
                   "format": format,
                   "stream": False
                   }
        response = requests.post(URL + COMPLETIONS, json=payload)
        if response.status_code != 200:
            raise Exception("Error: Server responded with ",
                            response.status_code)
        print("=====================CHATMODEL====================================")
        print(response.json())
        print("=========================================================")
        return response.json()

    def step(self, weights, combined_image): #weights, combined_image)

        responses = []
        buffered = BytesIO()
        combined_image.save(buffered, format="PNG")
        encoded_image = b64encode(buffered.getvalue()).decode("utf-8")

        # first message
        self.messages.extend(
            [
            {
                'role': 'system',
                'content': f"""

                "You are an assistant that fills in JSON fields with descriptive and elaborated answers. 
                For string fields like 'Driving_Scenario', 'Lane_Option', 'Parked_Vehicles', and 'Building_Proximity', 
                always provide a detailed explanation instead of a short word. Example: instead of 'Highway', say 
                'A three-lane highway with light traffic, smooth asphalt, and clear lane markings'."
                and help defining the weights of the following metrics:

                Safety Metrics:
                Weight_Collision: a function that increases collision penalty as the vehicle gets closer to an obstacle, so near obstacles have much higher risk than far ones. Initial: {weights['w_coll']:.2f}  
                Weight_Deviation: a penalty that increases as the vehicle moves further from the desired lane or path. Initial: {weights['w_dev']:.2f}  
                Weight_Distance: a penalty that increases when the vehicle’s distance to the goal becomes longer than necessary. Initial: {weights['w_dis']:.2f}  
                Weight_Speed: a penalty for speeds that are too high or too low compared to the desired speed profile. Initial: {weights['w_speed']:.2f}  

                Comfort Metrics:
                Weight_Lat: a penalty for high sideways (lateral) acceleration that could cause discomfort. Initial: {weights['w_lat']:.2f}  
                Weight_Lon: a penalty for high forward/backward (longitudinal) acceleration changes that could cause discomfort. Initial: {weights['w_lon']:.2f}  
                Weight_Cent: a penalty for high centripetal acceleration when turning, linked to cornering comfort. Initial: {weights['w_cent']:.2f}  
                """
            },
            {'role': 'user',
                'content': self.query[0],
                'images': [encoded_image],
             },
            ]
        )
        
        response = self.chat_model(format=q1.schema(), messages=self.messages)
        message = response['message']
        # print(message['content'])
        self.messages.append(message)
        responses.append(response)

        # second message
        self.messages.append({'role': 'user', 'content': self.query[1]})
        response = self.chat_model(format=q2.schema(), messages=self.messages)
        message = response['message']
        # print(message['content'])
        self.messages.append(message)
        responses.append(response)
      
        # third message
        self.messages.append({'role': 'user', 'content': self.query[2]})
        response = self.chat_model(format=q3.schema(), messages=self.messages)
        message = response['message']
        # print(message['content'])
        self.messages.append(message)
        responses.append(response)

        if len(self.messages) > self.memory*7:
            self.messages = self.messages[-self.memory*7:]

        return responses


class WeightScore:
    def __init__(self):
        self.w_coll = 1.5
        self.w_dev =  5.0
        self.w_dis =  2.5
        self.w_speed =  1.5
        self.w_lat =  4.5
        self.w_lon =  3.0
        self.w_cent =  3.5

    def update_weights(self, response):
        raw_json = response[-1]["message"]["content"]
        weights = json.loads(raw_json)
        self.w_coll = weights["Weight_Collision"]
        self.w_dev = weights["Weight_Deviation"]
        self.w_dis = weights["Weight_Distance"]
        self.w_speed = weights["Weight_Speed"]
        self.w_lat = weights["Weight_Lat"]
        self.w_lon = weights["Weight_Lon"]
        self.w_cent = weights["Weight_Cent"]
        
if __name__ == '__main__':
    from PIL import Image

    vlm = VLM()
    weights = WeightScore()

    image = Image.open("./test_images/im1.png")
    response = vlm.step(weights, image)
    weights.update_weights(response)
    print(response)
    print("\n\n")
    image = Image.open("./test_images/im2.png")
    response = vlm.step(weights, image)
    weights.update_weights(response)
    print(response)
    
    
