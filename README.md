# solhacks
# FairScore: AI-Powered Fairness Analysis for Loan Approvals

FairScore is an application developed to analyze fairness in loan approvals. It implements two approaches:
- **Equality of Opportunity:** Evaluates loan applications using only financial merit (ignoring protected characteristics).
- **Equality of Outcomes:** Adjusts decisions so that subgroups (e.g., "White Male", "Asian Female", "Hispanic Male", etc.) achieve similar approval rates by taking into account sensitive attributes.

The application also includes an interactive dashboard (built with Streamlit) and an AI chatbot that explains and discusses the graphs.

## Project Structure
## Prerequisites

- **Python 3.9+** is required.
- Git for cloning the repository.
- (Optional but recommended) A virtual environment to isolate dependencies.

## Setup Instructions

### 1. Clone the Repository

- Clone the repository from GitHub:
- git clone https://github.com/alpg00/solhacks.git
- cd solhacks

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

Create a virtual environment:

- bash
- Copy
- python -m venv venv

Activate the virtual environment:

- On macOS/Linux:

- bash
- Copy
- source venv/bin/activate

- On Windows:

- bash
- Copy
- venv\Scripts\activate

### 3. Install Dependencies
- Install the required packages:

- bash
- Copy
- pip install -r requirements.txt

The requirements.txt file should include (but is not limited to):

- openai

- python-dotenv

- streamlit

- pandas

- matplotlib

- seaborn

- numpy

- pywebview

### 4. Configure Environment Variables

Create a .env file in the project root (if it doesn’t exist) and add your OpenAI API key:

- ini
- Copy
- OPENAI_API_KEY=your_actual_openai_api_key_here

### 5. Run the Application

The main entry point is app.py. To launch the application, run:

- bash
- Copy
- python app.py --input data/bigdata.csv --rate 0.5 --output decisions.json
- --input: Path to the CSV file (e.g., data/bigdata.csv).

- --rate: Target approval rate (a number between 0.0 and 1.0).

- --output: Filename for saving the loan approval decisions (e.g., decisions.json).

This command will:

- Start the Streamlit dashboard.

- Open a desktop window using PyWebView.

- Process the loan application data using both fairness algorithms.

- Display visualizations (graphs and tables) of the approval statistics.

- Save the loan decisions to the specified JSON file.

For questions or issues, please contact Alp Gokcehan at alpg@ad.unc.edu.
