from pydantic import BaseModel
from instructor import from_openai
from openai import OpenAI
from typing import List
import random
import json
from datetime import datetime
import pandas as pd

pairs_to_generate = 20

# Define schema for single QA item
class QAItem(BaseModel):
    question: str
    answer: str
    equipment_problem: str
    tools_required: List[str]
    steps: List[str]
    safety_info: str
    tips: str

# Define schema for single QA item only

# Load OpenAI client
openai_client = OpenAI()

experts = [
    {
        "id": "appliance_repair",
        "details": "Expert home appliance repair technician with 20+ years of experience. Focus: Common household appliances (refrigerators, washing machines, dryers, dishwashers, ovens). Emphasis: Technical details and practical homeowner solutions."
    },
    {
        "id": "plumbing_repair", 
        "details": "Professional plumber with extensive residential experience. Focus: Common plumbing issues (leaks, clogs, fixture repairs, pipe problems). Emphasis: Safety for homeowner attempts and realistic solutions."
    },
    {
        "id": "electrical_repair",
        "details": "Licensed electrician specializing in safe home electrical repairs. Focus: SAFE homeowner-level electrical work (outlet replacement, switch repair, light fixture installation). Emphasis: Critical safety warnings and when to call professionals."
    },
    {
        "id": "hvac_maintenance",
        "details": "HVAC technician specializing in homeowner maintenance. Focus: Basic HVAC maintenance and troubleshooting (filter changes, thermostat issues, vent cleaning, basic troubleshooting). Emphasis: Seasonal considerations and maintenance best practices."
    },
    {
        "id": "general_home_repair",
        "details": "Skilled handyperson with general home repair expertise. Focus: Common general repairs and maintenance (drywall repair, door/window problems, flooring issues, basic carpentry). Emphasis: Material specifications and practical DIY solutions."
    }
]

# Wrap with Instructor
client = from_openai(openai_client)

# Generate 20 QA pairs with random expert selection
generated_pairs = []
used_experts = []

for i in range(pairs_to_generate):
    # Randomly select an expert
    selected_expert = random.choice(experts)
    used_experts.append(selected_expert['id'])
    
    # Create prompt for this specific expert
    prompt = f"""
Generate a synthetic DIY home repair Q&A pair as a {selected_expert['details']}

Create a realistic question and comprehensive answer that includes:
- Question: A specific, realistic homeowner problem
- Answer: Clear, actionable solution steps
- Equipment Problem: What's wrong with the equipment/system
- Tools Required: Specific tools needed (commonly available)
- Steps: Detailed, sequential repair steps
- Safety Info: Relevant safety warnings and precautions
- Tips: Practical advice and best practices

Focus on the expertise area: {selected_expert['id'].replace('_', ' ').title()}
"""
    
    # Generate single QA pair
    qa_item = client.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini", # gpt-5-mini is overly verbose
        response_model=QAItem,
    )
    
    generated_pairs.append(qa_item)
    
    # Print each generated pair
    print(f"\n=== QA PAIR {i+1} (Expert: {selected_expert['id']}) ===")
    print(f"Question: {qa_item.question}")
    print(f"Answer: {qa_item.answer}")
    print(f"Equipment Problem: {qa_item.equipment_problem}")
    print(f"Tools Required: {qa_item.tools_required}")
    print(f"Steps: {qa_item.steps}")
    print(f"Safety Info: {qa_item.safety_info}")
    print(f"Tips: {qa_item.tips}")
    print("-" * 50)

print(f"\nGenerated {len(generated_pairs)} QA pairs with random expert selection")

# Save generated pairs to JSON file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"qa_pairs_{timestamp}.json"

# Convert Pydantic models to dictionaries for JSON serialization
qa_data = []
for i, qa_item in generated_pairs:
    qa_dict = {
        "question": qa_item.question,
        "answer": qa_item.answer,
        "equipment_problem": qa_item.equipment_problem,
        "tools_required": qa_item.tools_required,
        "steps": qa_item.steps,
        "safety_info": qa_item.safety_info,
        "tips": qa_item.tips
    }
    qa_data.append(qa_dict)

# Save to JSON file
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(qa_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(generated_pairs)} QA pairs to {filename}")

# Create pandas DataFrame for failure analysis
print("\n=== CREATING FAILURE ANALYSIS DATAFRAME ===")

# Prepare data for DataFrame
df_data = []
for i, qa_item in enumerate(generated_pairs):
    row = {
        'trace_id': i + 1,
        'question': qa_item.question,
        'answer': qa_item.answer,
        'equipment_problem': qa_item.equipment_problem,
        'tools_required': qa_item.tools_required,
        'steps': qa_item.steps,
        'safety_info': qa_item.safety_info,
        'tips': qa_item.tips,
        # Binary failure mode columns (0 = success, 1 = failure)
        'incomplete_answer': 0,
        'safety_violations': 0,
        'unrealistic_tools': 0,
        'overcomplicated_solution': 0,
        'missing_context': 0,
        'poor_quality_tips': 0
    }
    df_data.append(row)

# Create DataFrame
df = pd.DataFrame(df_data)

print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Display basic info about the DataFrame
print(f"\nDataFrame shape: {df.shape}")
print(f"Failure mode columns initialized to 0 (success)")

# Save DataFrame to CSV for manual labeling
df_filename = f"qa_failure_analysis_{timestamp}.csv"
df.to_csv(df_filename, index=False)
print(f"DataFrame saved to {df_filename} for manual failure labeling")

# Display first few rows
print(f"\nFirst 3 rows of DataFrame:")
print(df.head(3).to_string())

