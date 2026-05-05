EVAL_CASES: list[dict] = [
    {
        "id": "t001",
        "scenario": "record_transaction",
        "input": "Today lunch 35 yuan",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 35,
            "direction": "expense",
            "category": "Food & Beverage",
        },
    },
    {
        "id": "t002",
        "scenario": "record_transaction",
        "input": "Coffee 28 yuan",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 28,
            "direction": "expense",
            "category": "Food & Beverage",
        },
    },
    {
        "id": "t003",
        "scenario": "record_transaction",
        "input": "Client paid 5000 to ICBC",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 5000,
            "direction": "income",
        },
    },
    {
        "id": "t004",
        "scenario": "record_transaction",
        "input": "Subway 3 yuan",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 3,
            "direction": "expense",
            "category": "Transportation",
        },
    },
    {
        "id": "t005",
        "scenario": "clarify",
        "input": "Record today's lunch",
        "expect": {
            "fn": "clarify",
            "missing_fields": ["amount_yuan"],
        },
    },
    {
        "id": "t006",
        "scenario": "generate_report",
        "input": "Generate this month's report by category",
        "expect": {
            "fn": "generate_report",
            "group_by": "category",
        },
    },
    {
        "id": "t007",
        "scenario": "generate_report",
        "input": "Show daily spending this month",
        "expect": {
            "fn": "generate_report",
            "group_by": "day",
        },
    },
    {
        "id": "t008",
        "scenario": "rag_retrieve",
        "input": "What is the travel reimbursement limit?",
        "expect": {
            "fn": "rag_retrieve",
        },
    },
    {
        "id": "t009",
        "scenario": "rag_retrieve",
        "input": "Business trip reimbursement rules",
        "expect": {
            "fn": "rag_retrieve",
        },
    },
    {
        "id": "t010",
        "scenario": "clarify",
        "input": "Record one",
        "expect": {
            "fn": "clarify",
        },
    },
]


SCENARIO_THRESHOLDS = {
    "record_transaction": 0.90,
    "generate_report": 0.85,
    "rag_retrieve": 0.85,
    "clarify": 0.92,
}
