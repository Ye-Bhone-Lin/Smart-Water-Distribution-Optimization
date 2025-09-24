# Water Condition Chatbot

This feature provides a command-line chatbot that delivers real-time water condition updates and insights using USGS data and Groq's LLM API.

## Features

- Fetches real-time water data (streamflow, gauge height) from USGS for a specified site.
- Uses Groq's LLM to generate insights and implications for water management.
- Interactive command-line interface.
- Customizable USGS site ID.
- system + user role prompt

## How It Works

1. The chatbot fetches water data from the USGS API for a given site.
2. For each variable (e.g., streamflow), it queries Groq's LLM for analysis and management insights.
3. Results are displayed interactively in the terminal.

## Requirements

- Python 3.x
- requests
- groq
- python-dotenv

Install dependencies with:

```bash
pip install requests groq python-dotenv
```

## Setup

1. Obtain a Groq API key and add it to a `.env` file:

    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```

2. (Optional) Change the USGS site ID in `water_chat.py` to target a different location.

## Usage

Run the chatbot from the command line:

```bash
python water_chat.py
```

Type your questions or press Enter to get the latest water update. Type `quit` to exit.

## File Structure

- `water_chat.py`: Main chatbot script.

## Notes

- The chatbot streams LLM responses for a more interactive experience.
- USGS site IDs can be found at https://waterdata.usgs.gov/nwis.
- Make sure your API key is valid and your environment is configured.
