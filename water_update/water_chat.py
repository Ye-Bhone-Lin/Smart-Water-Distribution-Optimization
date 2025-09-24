import requests
import datetime
from groq import Groq
from dotenv import load_dotenv
import os 

load_dotenv()

def get_usgs_water_data(site_id="01646500"):
    """
    Fetch real-time water data from USGS for a given site.
    site_id: USGS gauge ID (default: Potomac River at Point of Rocks, MD)
    """
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_id}&parameterCd=00060,00065&siteStatus=all"
    resp = requests.get(url)
    if resp.status_code != 200:
        return f"Error fetching data: {resp.status_code}"
    
    data = resp.json()
    results = {}
    for ts in data["value"]["timeSeries"]:
        variable = ts["variable"]["variableName"]
        if ts["values"] and ts["values"][0]["value"]:
            value = ts["values"][0]["value"][-1]["value"]
            time = ts["values"][0]["value"][-1]["dateTime"]
            results[variable] = (value, time)
    return results

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)
def chatbot():
    print("💧 Water Condition Chatbot (type 'quit' to exit)")
    while True:
        user = input("You: ")
        if user.lower() == "quit":
            print("Bot: Goodbye!")
            break

        site = "01646500"  # Change this to any USGS site ID as needed
        data = get_usgs_water_data(site)

        if isinstance(data, dict):
            print("Bot: Here is the latest water update:")
            for var, (val, ts) in data.items():
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides real-time water condition updates based on USGS data."
                        },
                        {
                            "role": "user",
                            "content": f"Water condition update and analysis: {var} is {val} at {ts}. Provide insights and implications for water management."
                        },
                    ],
                    temperature=1,
                    max_completion_tokens=8192,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=True,
                    stop=None
                )

                print("Insights:", end=" ")
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)

        else:
            print("Bot:", data)

if __name__ == "__main__":
    chatbot()