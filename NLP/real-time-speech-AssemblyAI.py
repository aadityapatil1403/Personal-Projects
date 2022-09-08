import pyaudio
import websockets
import asyncio
import base64
import json

auth_key = "2c3ac8b5204e46a889fded05b3236995"

#constants to create the stream (using recommended values)
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

#start recording
stream = p.open(
	format=FORMAT,
	channels=CHANNELS,
	rate=RATE,
	input=True,
	frames_per_buffer=FRAMES_PER_BUFFER
)

#URL for connection to AssemblyAI
URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

#creating asynchronous send/receieve functions
async def send_receive():
   print("Connecting websocket to url {}".format(URL))

   #creating connection(_ws)
   async with websockets.connect(
       URL,
       extra_headers=(("Authorization", auth_key),),
       ping_interval=5,
       ping_timeout=20
   ) as _ws:

       await asyncio.sleep(0.1)
       print("Receiving SessionBegins ...")

       session_begins = await _ws.recv()
       print(session_begins)
       print("Sending voice ...")

       #send function to get input from mic and send bit data to websocket
       async def send():
           while True:
               try:
                   data = stream.read(FRAMES_PER_BUFFER)
                   data = base64.b64encode(data).decode("utf-8")
                   json_data = json.dumps({"audio_data":str(data)})
                   await _ws.send(json_data)

               except websockets.exceptions.ConnectionClosedError as e:
                   print(e)
                   assert e.code == 4008
                   break

               except Exception as e:
                   assert False, "Not a websocket 4008 error"

               await asyncio.sleep(0.01)
          
           return True
      
      #receive function to get response from AssemblyAI
       async def receive():
           while True:
               try:
                   result_str = await _ws.recv()
                   if json.loads(result_str)['message_type'] == 'FinalTranscript':
                    print(json.loads(result_str)['text'])

               except websockets.exceptions.ConnectionClosedError as e:
                   print(e)
                   assert e.code == 4008
                   break
                
               except Exception as e:
                   assert False, "Not a websocket 4008 error"
      
      #wait to return result 
       send_result, receive_result = await asyncio.gather(send(), receive())

while True:
    asyncio.run(send_receive())
    