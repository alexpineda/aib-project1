from pydantic import BaseModel
from instructor import from_openai
from openai import OpenAI
from typing import List
import random
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        "details": "Expert home appliance repair technician with 20+ years of experience. Focus: Common household appliances (refrigerators, washing machines, dryers, dishwashers, ovens). Emphasis: Technical details and practical homeowner solutions.",
    },
    {
        "id": "plumbing_repair",
        "details": "Professional plumber with extensive residential experience. Focus: Common plumbing issues (leaks, clogs, fixture repairs, pipe problems). Emphasis: Safety for homeowner attempts and realistic solutions.",
    },
    {
        "id": "electrical_repair",
        "details": "Licensed electrician specializing in safe home electrical repairs. Focus: SAFE homeowner-level electrical work (outlet replacement, switch repair, light fixture installation). Emphasis: Critical safety warnings and when to call professionals.",
    },
    {
        "id": "hvac_maintenance",
        "details": "HVAC technician specializing in homeowner maintenance. Focus: Basic HVAC maintenance and troubleshooting (filter changes, thermostat issues, vent cleaning, basic troubleshooting). Emphasis: Seasonal considerations and maintenance best practices.",
    },
    {
        "id": "general_home_repair",
        "details": "Skilled handyperson with general home repair expertise. Focus: Common general repairs and maintenance (drywall repair, door/window problems, flooring issues, basic carpentry). Emphasis: Material specifications and practical DIY solutions.",
    },
]

# Check if JSON file exists and load from it
json_filename = "qa_pairs.json"
if os.path.exists(json_filename):
    print(f"Found existing JSON file: {json_filename}")
    print("Loading QA pairs from existing file...")

    with open(json_filename, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Convert back to QAItem objects
    generated_pairs = []
    for qa_dict in qa_data:
        # Remove trace_id if it exists (from failure analysis JSON)
        if "trace_id" in qa_dict:
            qa_dict = {k: v for k, v in qa_dict.items() if k != "trace_id"}
        qa_item = QAItem(**qa_dict)
        generated_pairs.append(qa_item)

    print(f"Loaded {len(generated_pairs)} QA pairs from {json_filename}")
else:
    print("No existing JSON file found. Generating new QA pairs...")

    # Wrap with Instructor
    client = from_openai(openai_client)

    # Generate 20 QA pairs with random expert selection
    generated_pairs = []
    used_experts = []

    for i in range(pairs_to_generate):
        # Randomly select an expert
        selected_expert = random.choice(experts)
        used_experts.append(selected_expert["id"])

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
            model="gpt-4o-mini",  # gpt-5-mini is overly verbose
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

# Save generated pairs to JSON file (only if we generated new data)
if not os.path.exists(json_filename):  # Only save if we didn't load from existing file
    # Convert Pydantic models to dictionaries for JSON serialization
    qa_data = []
    for qa_item in generated_pairs:
        qa_dict = {
            "question": qa_item.question,
            "answer": qa_item.answer,
            "equipment_problem": qa_item.equipment_problem,
            "tools_required": qa_item.tools_required,
            "steps": qa_item.steps,
            "safety_info": qa_item.safety_info,
            "tips": qa_item.tips,
        }
        qa_data.append(qa_dict)

    # Save to JSON file
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(generated_pairs)} QA pairs to {json_filename}")
else:
    print(f"\nUsing existing data from {json_filename}")

# Create pandas DataFrame for failure analysis
print("\n=== CREATING FAILURE ANALYSIS DATAFRAME ===")

# Prepare data for DataFrame
df_data = []
for i, qa_item in enumerate(generated_pairs):
    row = {
        "trace_id": i + 1,
        "question": qa_item.question,
        "answer": qa_item.answer,
        "equipment_problem": qa_item.equipment_problem,
        "tools_required": qa_item.tools_required,
        "steps": qa_item.steps,
        "safety_info": qa_item.safety_info,
        "tips": qa_item.tips,
        # Binary failure mode columns (0 = success, 1 = failure)
        "incomplete_answer": 0,
        "safety_violations": 0,
        "unrealistic_tools": 0,
        "overcomplicated_solution": 0,
        "missing_context": 0,
        "poor_quality_tips": 0,
    }
    df_data.append(row)

# Create DataFrame
df = pd.DataFrame(df_data)

print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

# Display first few rows
print(f"\nFirst 3 rows of DataFrame:")
print(df.head(3).to_string())

# failure_modes = [
#     "Incomplete Answer (incomplete_answer)",
#     "Safety Violations (safety_violations)",
#     "Unrealistic Tools (unrealistic_tools)",
#     "Overcomplicated Solution (overcomplicated_solution)",
#     "Missing Context (missing_context)",
#     "Poor Quality Tips (poor_quality_tips)"
# ]

# Define indepth failure detection prompts for each failure mode
failure_detection_prompts = {
    "incomplete_answer": """
Evaluate this DIY home repair Q&A pair for INCOMPLETE ANSWER failure mode.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

INCOMPLETE ANSWER failure occurs when:
- The answer doesn't fully address the question asked
- Critical steps are missing from the solution
- The answer is too brief or lacks necessary detail
- Important troubleshooting steps are omitted
- The solution doesn't cover all aspects of the problem

Rate this Q&A pair for INCOMPLETE ANSWER failure:
- 1 = FAILURE: Answer is incomplete, missing critical information
- 0 = SUCCESS: Answer is complete and comprehensive

Respond with only "1" or "0".
""",
    "safety_violations": """
Evaluate this DIY home repair Q&A pair for SAFETY VIOLATIONS failure mode.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

SAFETY VIOLATIONS failure occurs when:
- Dangerous or unsafe practices are recommended
- Critical safety warnings are missing
- Electrical work is suggested without proper safety measures
- Hazardous materials handling is not properly addressed
- Safety equipment is not mentioned when needed

Rate this Q&A pair for SAFETY VIOLATIONS failure:
- 1 = FAILURE: Contains safety violations or missing critical safety info
- 0 = SUCCESS: Proper safety measures are included and emphasized

Respond with only "1" or "0".
""",
    "unrealistic_tools": """
Evaluate this DIY home repair Q&A pair for UNREALISTIC TOOLS failure mode.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

UNREALISTIC TOOLS failure occurs when:
- Specialized or expensive tools are required that homeowners wouldn't have
- Professional-grade equipment is suggested for basic repairs
- Tools that require special training or certification are recommended
- Equipment that's not commonly available at hardware stores is listed

Rate this Q&A pair for UNREALISTIC TOOLS failure:
- 1 = FAILURE: Requires unrealistic or inaccessible tools
- 0 = SUCCESS: Uses common, accessible tools that homeowners can obtain

Respond with only "1" or "0".
""",
    "overcomplicated_solution": """
Evaluate this DIY home repair Q&A pair for OVERCOMPLICATED SOLUTION failure mode.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

OVERCOMPLICATED SOLUTION failure occurs when:
- The solution is unnecessarily complex for the problem
- Too many steps are required for a simple fix
- The approach is more complex than needed
- Professional-level techniques are suggested for basic repairs
- The solution doesn't match the skill level of a typical homeowner

Rate this Q&A pair for OVERCOMPLICATED SOLUTION failure:
- 1 = FAILURE: Solution is unnecessarily complex or overcomplicated
- 0 = SUCCESS: Solution is appropriately simple and straightforward

Respond with only "1" or "0".
""",
    "missing_context": """
Evaluate this DIY home repair Q&A pair for MISSING CONTEXT failure mode.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

MISSING CONTEXT failure occurs when:
- Important background information is missing
- The answer doesn't explain why certain steps are necessary
- Context about the problem or solution is lacking
- Assumptions are made without explanation
- The answer jumps into steps without proper setup

Rate this Q&A pair for MISSING CONTEXT failure:
- 1 = FAILURE: Missing important context or background information
- 0 = SUCCESS: Provides adequate context and background

Respond with only "1" or "0".
""",
    "poor_quality_tips": """
Evaluate this DIY home repair Q&A pair for POOR QUALITY TIPS failure mode.

Question: {question}
Answer: {answer}
Equipment Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

POOR QUALITY TIPS failure occurs when:
- Tips are generic, unhelpful, or obvious
- Tips don't add value to the solution
- Tips are poorly written or unclear
- Tips contain incorrect information
- Tips are not relevant to the specific problem

Rate this Q&A pair for POOR QUALITY TIPS failure:
- 1 = FAILURE: Tips are poor quality, unhelpful, or incorrect
- 0 = SUCCESS: Tips are helpful, relevant, and well-written

Respond with only "1" or "0".
""",
}


def detect_failure_mode(qa_item: QAItem, failure_mode: str, model: str = "gpt-5-nano"):
    """
    Query a small model to detect if a Q&A pair has a specific failure mode.
    Returns 1 for failure, 0 for success.
    """
    prompt_template = failure_detection_prompts[failure_mode]

    # Format the prompt with the Q&A data
    prompt = prompt_template.format(
        question=qa_item.question,
        answer=qa_item.answer,
        equipment_problem=qa_item.equipment_problem,
        tools_required=", ".join(qa_item.tools_required),
        steps="\n".join([f"{i+1}. {step}" for i, step in enumerate(qa_item.steps)]),
        safety_info=qa_item.safety_info,
        tips=qa_item.tips,
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
        )

        result = response.choices[0].message.content.strip()
        return int(result) if result in ["0", "1"] else 0

    except Exception as e:
        print(f"Error detecting {failure_mode} for trace {qa_item}: {e}")
        return 0


# Define failure mode columns
failure_mode_columns = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]

# Check if failure analysis JSON already exists
failure_analysis_json = "qa_failure_analysis.json"
if os.path.exists(failure_analysis_json):
    print(f"\nFound existing failure analysis file: {failure_analysis_json}")
    print("Loading existing failure analysis data into dataframe...")

    # Load the entire JSON file into the dataframe
    df = pd.read_json(failure_analysis_json)

    print("Loaded existing failure analysis data into dataframe")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
else:
    # Update dataframe with failure detection results
    print("\n=== DETECTING FAILURE MODES ===")

    for idx, row in df.iterrows():
        print(f"Processing trace {row['trace_id']}...")

        # Create QAItem object for this row
        qa_item = QAItem(
            question=row["question"],
            answer=row["answer"],
            equipment_problem=row["equipment_problem"],
            tools_required=row["tools_required"],
            steps=row["steps"],
            safety_info=row["safety_info"],
            tips=row["tips"],
        )

        # Detect each failure mode
        for failure_mode in failure_mode_columns:
            failure_score = detect_failure_mode(qa_item, failure_mode)
            df.at[idx, failure_mode] = failure_score
            print(f"  {failure_mode}: {failure_score}")

    # Save updated dataframe
    df.to_json(failure_analysis_json, orient="records", indent=2)
    print(f"\nUpdated dataframe saved to {failure_analysis_json}")

# Display summary statistics
print("\n=== FAILURE MODE SUMMARY ===")
for failure_mode in failure_mode_columns:
    failure_count = df[failure_mode].sum()
    total_count = len(df)
    percentage = (failure_count / total_count) * 100
    print(f"{failure_mode}: {failure_count}/{total_count} ({percentage:.1f}%)")

# Create matplotlib visualizations for failure analysis
print("\n=== CREATING MATPLOTLIB VISUALIZATIONS ===")


# Set up the plotting style
plt.style.use("default")
sns.set_palette("husl")

# Create a simple figure with just the heatmap
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle(
    "DIY Home Repair Q&A Failure Analysis Heatmap", fontsize=16, fontweight="bold"
)

# Failure Mode Heatmap
heatmap_data = df[failure_mode_columns].values
im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")
ax.set_title("Failure Mode Heatmap\n(Red=Failure, Green=Success)", fontweight="bold")
ax.set_xlabel("Failure Modes")
ax.set_ylabel("Sample ID")
ax.set_xticks(range(len(failure_mode_columns)))
ax.set_xticklabels(
    [col.replace("_", "\n").title() for col in failure_mode_columns],
    rotation=45,
    ha="right",
)
ax.set_yticks(range(0, len(df), 2))
ax.set_yticklabels(range(1, len(df) + 1, 2))

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Failure Status", rotation=270, labelpad=15)

# Adjust layout and save
plt.tight_layout()

# Save the visualization
output_filename = "qa_failure_analysis_dashboard.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Visualization saved as: {output_filename}")

# Display the plot
plt.show()
# Find samples with multiple failures
multi_failure_samples = df[df[failure_mode_columns].sum(axis=1) > 1][
    "trace_id"
].tolist()
if multi_failure_samples:
    print(f"Samples with multiple failures: {multi_failure_samples}")
else:
    print("No samples had multiple failure modes")

# Find perfect samples (no failures)
perfect_samples = df[df[failure_mode_columns].sum(axis=1) == 0]["trace_id"].tolist()
print(f"Perfect samples (no failures): {perfect_samples}")
