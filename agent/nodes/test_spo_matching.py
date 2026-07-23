"""Test du state_matching_node sur job_application (run_17, Protocole 3) -- les deux variantes
de state_space demandees."""

from agent.nodes.state_matching_node import match_spo_to_u

SPO = [
    ("Write job application", "follows", "Send job application"),
    ("Send job application", "follows", "Report job application"),
    ("Report job application", "follows", "Recieve potential job offer"),
    ("Report job application", "follows", "Receive confirmation of application entry"),
    ("Receive confirmation of application entry", "follows", "Feedback on application received"),
    ("Negotiate job interview", "follows", "Start job (probation phase)"),
    ("Negotiate job interview", "follows", "Write job application"),
    ("Start job (probation phase)", "follows", "Probation phase finished"),
    ("Probation phase finished", "follows", "Rate company"),
    ("Rate company", "follows", "Receive feedback on interview"),
    ("Receive feedback on interview", "follows", "Report feedback"),
    ("Report feedback", "follows", "Application process finished"),
    ("Report feedback", "follows", "Receive potential job offers"),
    ("Report feedback", "follows", "Unemployed"),
    ("Recieve potential job offer", "follows", "Application process finished"),
    ("Recieve potential job offer", "follows", "Receive potential job offers"),
    ("Recieve potential job offer", "follows", "Unemployed"),
    ("Recieve potential job offer", "follows", "Recieve potential job offer"),
    ("Unemployed", "follows", "Write job application"),
    ("Receive potential job offers", "follows", "Application process finished"),
    ("Feedback on application received", "follows", "Negotiate job interview"),
    ("Feedback on application received", "follows", "Write job application"),
]

STATE_SPACE_ZERO_SHOT = {
    "job_applications": ["reported", "confirmed", "rated"],
    "potential_job_offers": ["sent", "received", "continued"],
    "job_interview": ["negotiated"],
    "probation_phase": ["entered", "completed"],
    "reviews": ["seen"],
    "job": ["permanent", "ended"],
    "process": ["ended"],
}

STATE_SPACE_ONE_SHOT = {
    "job_application": ["confirmed", "rated", "reported"],
    "job_offer": ["sent", "received"],
    "job_interview": ["negotiated"],
    "probation_phase": ["entered"],
    "review": ["visible"],
    "job": ["permanent"],
    "process": ["ended"],
}

if __name__ == "__main__":
    for label, state_space in [("zero_shot", STATE_SPACE_ZERO_SHOT), ("one_shot", STATE_SPACE_ONE_SHOT)]:
        print(f"\n===== {label} =====")
        matches, graph = match_spo_to_u(SPO, state_space)
        for activity, m in matches.items():
            print(f"{activity!r:45} -> {m['match']:35} ({m['score']})")
        print(f"\nGraphe process ({len(graph)} aretes) :")
        for edge in graph:
            print(f"  {edge['from']} -> {edge['to']}")