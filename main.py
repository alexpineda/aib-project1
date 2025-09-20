from pydantic import BaseModel
from instructor import from_hf
from transformers import pipeline
from typing import List
import random

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

# Load HF model
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

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
client = from_hf(pipe)

# Generate 20 QA pairs with random expert selection
generated_pairs = []

for i in range(pairs_to_generate):
    # Randomly select an expert
    selected_expert = random.choice(experts)
    
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
    qa_item = client(
        prompt,
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

