# evaluation_data.py
# MODIFIED: Ground truth answers are now more concise to better match
# the expected output of an extractive QA model.

ground_truth_dataset = [
    {
        "question": "What is the validity period of the Battery Swapping Policy?",
        "ground_truth": "till 31st March 2025"
    },
    {
        "question": "What is the GST on Lithium-ion batteries?",
        "ground_truth": "18%"
    },
    {
        "question": "What is the price of a Tesla?",
        "ground_truth": "I do not have enough information to answer this question."
    }
]
