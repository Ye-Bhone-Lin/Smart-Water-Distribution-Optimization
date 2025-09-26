import requests
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class WaterDataBot:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)

    def fetch_usgs_data(self, site_id="01646500"):
        """
        Fetch real-time water data from USGS.
        Returns dict {variable_name: (value, timestamp)} or error string.
        """
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_id}&parameterCd=00060,00065&siteStatus=all"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = {}
            for ts in data.get("value", {}).get("timeSeries", []):
                variable = ts["variable"]["variableName"]
                if ts["values"] and ts["values"][0]["value"]:
                    value = ts["values"][0]["value"][-1]["value"]
                    time = ts["values"][0]["value"][-1]["dateTime"]
                    results[variable] = (value, time)
            return results
        except requests.RequestException as e:
            return f"Error fetching USGS data: {e}"

    def generate_insights(self, variable, value, timestamp):
        """
        Generate insights using Groq chat model.
        """
        try:
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant providing real-time water condition insights based on USGS data."
                    },
                    {
                        "role": "user",
                        "content": f"Water condition update and analysis: {variable} is {value} at {timestamp}. Provide insights and implications for water management."
                    },
                ],
                temperature=1,
                top_p=1,
                reasoning_effort="medium"
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {e}"
