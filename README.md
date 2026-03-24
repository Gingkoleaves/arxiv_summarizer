# ArXiv Paper Summarizer

This repository provides a Python script to fetch and summarize research papers from arXiv using the free Gemini API. Additionally, it demonstrates how to automate the extraction and summarization of arXiv articles daily based on specific keywords (see the section titled "Automatic Daily Extraction and Summarization" below). The tool is designed to help researchers, students, and enthusiasts quickly extract key insights from arXiv papers without manually reading through lengthy documents.

## Features

- **Single URL Summarization**: Summarize a single arXiv paper by providing its URL.
- **Batch URL Summarization**: Summarize multiple arXiv papers by listing their URLs in a text file.
- **Batch Keywords Summarization**: Fetch and summarize all papers from arXiv based on keywords and date ranges.
- **Easy Setup**: Simple installation and configuration process using Conda and pip.
- **Gemini API Integration**: Leverages the free Gemini API for high-quality summarization.

## Prerequisites

- Python 3.11
- Conda (for environment management)
- A Gemini API key (free to obtain)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shaier/arxiv_summarizer.git
cd arxiv_summarizer
```

### 2. Set Up the Conda Environment

Create and activate a Conda environment with Python 3.11:

```bash
conda create -n arxiv_summarizer python=3.11
conda activate arxiv_summarizer
```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Obtain your Gemini API key from [Google's Gemini API page](https://ai.google.dev/gemini-api/docs/api-key). Once you have the key, open the `url_summarize.py` file and replace `YOUR_GEMINI_API_KEY` on line 5 with your actual API key.

## Usage

### Summarize a Single Paper (Based on a Single URL)

To summarize a single arXiv paper, run the script and provide the arXiv URL (ensure it is the abstract page, not the PDF link):

```bash
python url_summarize.py
```

When prompted:

1. Enter `1` to summarize a single paper.
2. Provide the arXiv URL (e.g., `https://arxiv.org/abs/2410.08003`).

### Summarize Multiple Papers (Based on Multiple URLs)

To summarize multiple papers:

1. Add the arXiv URLs to the `links.txt` file, with one URL per line.
2. Run the script:

```bash
python url_summarize.py
```

3. When prompted, enter `2` to process all URLs listed in `links.txt`. Summaries are saved in `result.txt`.

## Example

Here’s an example of how to use the script:

```bash
python url_summarize.py
> Enter 1 for single paper or 2 for multiple papers: 1
> Enter the arXiv URL: https://arxiv.org/abs/2410.08003
```

### Summarize Multiple Papers (Based on Keywords)

`keywords_summarizer.py` enables fetching and summarizing papers based on specified keywords and date ranges. This is useful for tracking new research trends, generating related work sections, or conducting systematic reviews across multiple keywords at once.

### Usage

1. **Run the script** and provide your search criteria:

```bash
python keywords_summarizer.py
```

2. **Specify keywords and a date range** when prompted. Example input:

```bash
Enter keywords: "transformer, sparsity, MoE"
Enter start date (YYYY-MM-DD): 2017-01-01
Enter end date (YYYY-MM-DD): 2024-03-01
```

3. The script fetches relevant papers from arXiv and generates summaries. The results are saved in `result.txt`.

## Automatic Daily Extraction and Summarization

You can automate the extraction and summarization of arXiv articles based on specific keywords using Google Apps Script.  
 This setup will run daily and add newly found article titles (with links and summaries) to a Google Doc.

### Steps to Set Up

1.  **Open Google Apps Script**
    - Log in to your Google account and go to [Google Apps Script](https://script.google.com/home/my).
    - Click on **"New project"** in the top left.

2.  **Create a Google Doc**
    - Open [Google Docs](https://docs.google.com).
    - Click **Blank document** to create a new document.
    - Copy the **document ID** from the URL.
      - The ID is the long string in the document's URL, e.g., `123HEM4h5aQwygDk_A-xNaJ8CUoyMZTFsChyMk`.

3.  **Copy and Modify the Script**
    - Open the `daily_arxiv.txt` file in this repository.
    - Copy and paste its content into the Google Apps Script editor.
    - Locate the `var docId` in the script (around line 3) and replace it with the **Google Doc ID** from Step 2.
    - Add your **Gemini API Key** around **line 81** (look for `var apiKey =`).
    - Locate `var keywords = [...]` around **line 4** and update it with your preferred keywords.

4.  **Test the Script**
    - Click the **Run** button at the top to execute the script (you might need to provide permissions).
    - If everything works correctly, your Google Doc should now contain a list of arXiv article titles with links.

5.  **Schedule Daily Execution**
    - Click on the **clock icon** on the left (Triggers).
    - Click **"Add trigger"** in the bottom right.
    - Configure the trigger settings:
      - **Function**: Select the main function from the dropdown.
      - **Event Source**: Choose **Time-driven**.
      - **Type**: Select **Day timer**.
      - **Time Range**: Pick a time slot (e.g., midnight to 1 AM).
      - **Notifications**: Enable email notifications if you want updates.
    - Click **Save**.

Now, your script will automatically fetch and summarize new arXiv articles daily based on your chosen keywords!

### Setup Locally

Files structure:

┌────────────────────────┬─────────────────────────────────────────────┐
│ File │ Purpose │
├────────────────────────┼─────────────────────────────────────────────┤
│ topic_deep_analyzer.py │ Main analysis script │
├────────────────────────┼─────────────────────────────────────────────┤
│ setup_env.bat │ One-time conda env setup │
├────────────────────────┼─────────────────────────────────────────────┤
│ analyze.bat │ Wrapper (direct args or interactive wizard) │
└────────────────────────┴─────────────────────────────────────────────┘

Key arguments

--topic / -t Search keywords (required)
--start / -s Start date YYYY-MM-DD [default: 2024-01-01]
--end / -e End date YYYY-MM-DD [default: today]
--max / -n Max papers, 0=unlimited [default: 20]
--lang / -l en (English) or zh (Chinese) [default: en]
--pdf Render all output to PDF via pandoc
--no-download Use abstract text only, skip PDF download
--no-synthesis Skip the cross-paper synthesis report

Usage examples

:: English report, 20 papers
analyze.bat --topic "diffusion model" --start 2024-01-01 --max 20

:: Chinese report with PDF rendering
analyze.bat --topic "diffusion model" --start 2024-01-01 --lang zh --pdf

:: No synthesis, English
analyze.bat --topic "NeRF scene reconstruction" --max 10 --no-synthesis

PDF rendering note

--pdf requires pandoc + a LaTeX engine:

- Pandoc: https://pandoc.org/installing.html
- MiKTeX (LaTeX for Windows): https://miktex.org/
- For --lang zh, also needs SimSun font (included in Windows by default)

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## Support

If you encounter any issues or have questions, feel free to open an issue.
