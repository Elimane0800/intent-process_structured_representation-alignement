from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text_maternity = "Notify Social Security, Company in time. Gather information from companies."
text_job_app = (
    "Companies have to confirm that they received job applications and rate the application."
)

labels = [
    "an organization or actor that performs actions on other things",
    "a record, document, or case that is created, tracked, and updated over time",
]

for text, candidates in [
    (text_maternity, ["company"]),
    (text_job_app, ["company", "application"]),
]:
    for candidate in candidates:
        hypothesis_text = f"In this context, '{candidate}' refers to {{}}"
        result = classifier(text, labels, hypothesis_template=hypothesis_text)
        print(candidate, "->", dict(zip(result["labels"], [round(s, 3) for s in result["scores"]])))