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
    {
        "id": "t011",
        "scenario": "record_batch",
        "input": "Today breakfast 12 lunch 35 taxi 45",
        "expect": {
            "fn": "record_batch",
            "count": 3,
        },
    },
    {
        "id": "t012",
        "scenario": "record_batch",
        "input": "Groceries 20 taxi 15 coffee 30",
        "expect": {
            "fn": "record_batch",
            "count": 3,
        },
    },
    {
        "id": "t013",
        "scenario": "query_transactions",
        "input": "Show this month's food expenses",
        "expect": {
            "fn": "query_transactions",
            "direction": "expense",
            "category": "Food & Beverage",
        },
    },
    {
        "id": "t014",
        "scenario": "analyze_spending",
        "input": "Compare this month's dining spending with last month",
        "expect": {
            "fn": "analyze_spending",
            "compare_with": "prev_period",
        },
    },
    {
        "id": "t015",
        "scenario": "analyze_spending",
        "input": "How much did we spend each day in the last 7 days",
        "expect": {
            "fn": "analyze_spending",
            "period": "last_7_days",
            "group_by": "day",
        },
    },
    {
        "id": "t016",
        "scenario": "check_budget",
        "input": "How much food budget is remaining",
        "expect": {
            "fn": "check_budget",
            "category": "Food & Beverage",
        },
    },
    {
        "id": "t017",
        "scenario": "check_budget",
        "input": "Show overall budget status",
        "expect": {
            "fn": "check_budget",
        },
    },
    {
        "id": "t018",
        "scenario": "record_batch",
        "input": "Morning subway 3, lunch 18, afternoon coffee 25",
        "expect": {
            "fn": "record_batch",
            "count": 3,
        },
    },
]


SCENARIO_THRESHOLDS = {
    "record_transaction": 0.90,
    "record_batch": 0.85,
    "query_transactions": 0.85,
    "generate_report": 0.85,
    "analyze_spending": 0.85,
    "check_budget": 0.85,
    "rag_retrieve": 0.85,
    "clarify": 0.92,
}
