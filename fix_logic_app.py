import json

# Load current workflow
with open("workflow_current.json", "r", encoding="utf-8") as f:
    workflow = json.load(f)

# Remove Send_Final_Response (not Send_Initial_Response!)
if "Send_Final_Response" in workflow["properties"]["definition"]["actions"]:
    del workflow["properties"]["definition"]["actions"]["Send_Final_Response"]
    print("OK - Removed Send_Final_Response")
else:
    print("SKIP - Send_Final_Response already removed")

# Save fixed workflow
with open("workflow_fixed.json", "w", encoding="utf-8") as f:
    json.dump(workflow, f, indent=2, ensure_ascii=False)
    print("OK - Fixed workflow saved to workflow_fixed.json")
