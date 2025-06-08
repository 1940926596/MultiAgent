import os
import sys

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_tools.open_ai.base_agent_openai import BaseFinanceAgent
from agent_tools.data_handler.finance_report.macro_summary import macro_summary_function

import pandas as pd

class MacroFinanceAgent(BaseFinanceAgent):
    def __init__(self, name="MacroAgent", role="Analyze quarterly reports and summarize macro-level financial health.",
                 model="gpt-4", memory_size=8):
        super().__init__(name=name, role=role, model=model, function_schema=[macro_summary_function], memory_size=memory_size)
        self.interested_fields = ["form_type", "all_fields", "date", "tic"]

# Load dataset
csv_path = '../../../datasets/processed/financial_final.csv'
df = pd.read_csv(csv_path)

# Drop rows with missing required fields
required_fields = ["date", "tic", "form_type", "all_fields"]
df = df.dropna(subset=required_fields)

macro_agent = MacroFinanceAgent()

# Initialize columns for the results
df['macro_summary'] = None
df['risk_tags'] = None
df['macro_score'] = None

results = []

for idx, row in df.iterrows():
    input_data = {
        "date": row["date"],
        "tic": row["tic"],
        "form_type": row["form_type"],
        "all_fields": row["all_fields"]
    }
    
    # Skip rows with empty financial report text
    if not row["all_fields"] or row["all_fields"].strip() == "":
        print(f"Skipping index {idx} due to missing financial report.")
        continue
    
    try:
        prompt = (
            f"Date: {input_data['date']}\n"
            f"Form Type: {input_data['form_type']}\n"
            "Please summarize the operational status, potential risks, "
            "and provide a macro score for this quarter based on the financial report.\n"
            "Please respond by calling the 'macro_summary' function according to the specified schema."
        )
        
        result = macro_agent.ask_model(prompt)
        
        # Write results into DataFrame columns
        df.at[idx, 'macro_summary'] = result.get('summary', '')
        df.at[idx, 'risk_tags'] = ', '.join(result.get('risk_tag', []))
        df.at[idx, 'macro_score'] = result.get('macro_score', None)
        
        # Store results separately
        results.append({
            "date": row["date"],
            "tic": row["tic"],
            "summary": result.get("summary"),
            "risk_tag": result.get("risk_tag"),
            "macro_score": result.get("macro_score")
        })

    except Exception as e:
        print(f"[ERROR] Failed at index {idx}: {e}")

# Save the updated DataFrame with macro summaries
df.to_csv('../../../datasets/processed/financial_final_with_macro.csv', index=False)

# Save just the extracted macro summaries to a separate CSV
result_df = pd.DataFrame(results)
output_path = 'macro_summary_results.csv'
result_df.to_csv(output_path, index=False)
print(f"Saved macro summaries to {output_path}")
