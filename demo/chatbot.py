import streamlit as st
import requests
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Function to fetch USGS water data
def get_usgs_water_data(site_id="01646500"):
    """
    Fetch real-time water data from USGS for a given site.
    """
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={site_id}&parameterCd=00060,00065&siteStatus=all"
    resp = requests.get(url)
    if resp.status_code != 200:
        return f"Error fetching data: {resp.status_code}"
    
    data = resp.json()
    results = {}
    for ts in data.get("value", {}).get("timeSeries", []):
        variable = ts["variable"]["variableName"]
        if ts["values"] and ts["values"][0]["value"]:
            value = ts["values"][0]["value"][-1]["value"]
            time = ts["values"][0]["value"][-1]["dateTime"]
            results[variable] = (value, time)
    return results

# Streamlit app
st.title("💧 USGS Water Condition Chatbot")
st.write("Enter a USGS site ID to get the latest water conditions and insights.")

site_id = st.text_input("Site ID", value="01646500")

if st.button("Get Water Update"):
    if not site_id:
        st.warning("Please enter a valid site ID.")
    else:
        st.info("Fetching data...")
        data = get_usgs_water_data(site_id)
        
        if isinstance(data, dict) and data:
            st.success("Here is the latest water update:")
            for var, (val, ts) in data.items():
                st.write(f"**{var}**: {val} at {ts}")

                # Generate insights using Groq API
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
                    max_completion_tokens=1024,
                    top_p=1,
                    reasoning_effort="medium"
                )

                insights = completion.choices[0].message.content
                st.markdown(f"**Insights:** {insights}")
        else:
            st.error(f"Could not fetch data: {data}")
