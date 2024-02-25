from flask import Flask, jsonify
import whisper
import re
import json


import os
from openai import OpenAI
from dotenv import load_dotenv
import time
from moviepy.editor import *

from vid_2_audio import convert

import re
from math import sqrt
app = Flask(__name__)






load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


input_file = 'test.m4a'

temp = input_file.split('.')[0]
if 'mp4' in input_file:
    
    print(temp)
    audio_file = f"{temp}.mp3"  # Define the path for the extracted audio

    if os.path.exists(audio_file):
        print('found')
        input_file = audio_file
    else:
        print('converting')
        convert(input_file)
        input_file = audio_file



grade_prompt = """
I will provide you audio transcript from a presentation, and I want you to analyze it against the criteria below: 

Criteria:
- Pace: speaking to fast or slow, use the time stamps next to each line
- Usage of filler words: Examples being 'um', 'very', 'so', etc
- Content Organization: How well the speech is structured into introduction, body, and conclusion.
- Clarity and Precision: The use of clear, concise language and specific terms.
- Rhetorical Techniques: Use of persuasive elements, storytelling, analogies, and rhetorical questions.
- Language: Appropriateness of language, word choice, how interesting or engaging the presentation is.
- Conclusion: How well did the speaker conclude the presentation, for example a call to action

Asses each criteria from 0 to 10.

Example of a bad presentation:
4.86s - 11.16s:  So, imagine a world of personal computing interfaces tailored to the individual as
11.16s - 16.36s:  opposed to a single solution needing to appease two billion users both paid and unpaid.
20.06s - 23.34s:  And so who might I be to suggest something like that?
23.92s - 27.98s:  I've become known as a bit of a fixture in the Windows ecosystem over the past five years
27.98s - 32.06s:  through work with communities, technical bugs, and interface design.
33.0s - 38.32s:  I was privileged enough to learn the artistry of computing from some of the masters at Microsoft.
43.98s - 47.56s:  Recall a time when you couldn't put your hand on a specific document that you knew was there,
47.64s - 53.62s:  your desktop was overloaded with Windows, or your mother called you frantically because
53.62s - 57.4s:  she couldn't figure out how to log off from her computer on her overpopulated start menu.
57.52s - 57.96s:  .
59.22s - 69.16s:  Imagine a system with workspaces, built-in cloud synchronization, SQL on the desktop
69.16s - 75.22s:  with natural language processing assistance, and most importantly, form of use integrated
76.42s - 76.86s:  in function.

Example Rating:
- Pace: 4/10. The speaker talks too slow with lots of pauses (as deduced from the timestamps)
- Usage of filler words: 6/10. The presentation starts with 'So' and has many pauses.
- Content Organization: 5/10. Poor flow of information, try starting with the user story, instead of having it in the middle.
- Clarity and Precision: 4/10. It is not clear what the speaker is talking about, what are the listeners supposed to takeaway?
- Rhetorical Techniques: 6/10. There is an attempt to use storytelling however, it falls short in its significance.
- Language: 7/10. Language is appropriately technical while not being over detailed 
- Conclusion: 3/10. Cannot tell when the conclusion begins, there is no summary or takeaway

END OF BAD EXAMPLE.


Example of a good presentation:
11.96s - 15.8s:  Over the past two years youth unemployment has been on the rise. It
15.8s - 19.86s:  currently represents just under 40% of all unemployment in Australia. Young
19.86s - 23.9s:  graduates are leaving university and finding it more and more difficult to enter any
23.9s - 29.32s:  form of creative industry. Now the common stipulation is that you can't get a job
29.32s - 34.44s:  without experience but you can't get a job to get the experience. Now meanwhile
34.44s - 41.44s:  42% of small businesses failed in 2003 to 2007 and the figures haven't improved
41.44s - 46.56s:  much. Amongst many reasons this is happening is a consistent lack of
46.56s - 51.26s:  quality in their branding, marketing, websites and designs. The kind of
51.26s - 56.2s:  training that these graduates have just spent three to six years training for.
56.6s - 59.26s:  Now what if there was an enterprise that
59.26s - 63.96s:  bridged these two sets of frightening statistics? I want to build that bridge.
 
Example Rating:
- Pace: 8/10. The speaker maintains a steady pace throughout the presentation. However, there are moments where the speaker could slow down to allow the audience to absorb the information.
- Usage of filler words: 9/10. The speaker uses very few filler words, which helps maintain the flow and clarity of the presentation.
- Content Organization: 8/10. The speaker presents a clear problem and proposes a solution at the end. However, the transition between the problem and the solution could be smoother.
- Clarity and Precision: 8/10. The speaker uses clear and concise language. However, the speaker could use more specific terms to describe the problem and the proposed solution.
- Rhetorical Techniques: 7/10. The speaker uses statistics to emphasize the problem and proposes a solution in the form of a rhetorical question. However, the speaker could use more storytelling or analogies to make the presentation more engaging.
- Language: 8/10. The speaker uses appropriate language and word choice. However, the presentation could be more engaging with the use of more vivid language.
- Conclusion: 8/10. The speaker concludes the presentation with a call to action and leaves the listener wanting to hear more. However, the conclusion could be stronger with a more detailed description of the proposed solution.%

END OF GOOD EXAMPLE.

Here is the presentation audio

{video_text}

Rating:
"""


def gpt(prompt):
    stream = client.chat.completions.create(
                # model="gpt-4-0125-preview",
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.00
            )
    
    response= ''
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
        response+=str(chunk.choices[0].delta.content)
        # response = stream.choices[0].message.content

    return response



    


# speech_file = 'test.m4a'

result = whisper.transcribe(input_file, path_or_hf_repo="large", word_timestamps = True, initial_prompt='focus on filler words and unfinished words')
# print(video_text['text'])
video_text = ''
for segment in result['segments']:
    video_text +=f"{round(segment['start'], 2)}s - {round(segment['end'], 2)}s: {segment['text']}\n"
    # print(f"{round(segment['start'], 2)}s - {round(segment['end'], 2)}s: {segment['text']}")


print(video_text,  '\n')



grade = gpt(prompt=grade_prompt.format(video_text=video_text))

pattern = r'- ([\w\s]+): (\d+)/10'

matches = re.findall(pattern, grade)

# Convert the matches to a dictionary
results = {label.strip(): int(score) for label, score in matches}

# Convert the dictionary to JSON
json_output = json.dumps(results, indent=4)


@app.route('/metrics', methods=['GET'])
def respond_with_json():
    data = json_output
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
