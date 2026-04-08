"""
eval_dataset.py — Golden Dataset for RAG Evaluation

A golden dataset is a set of questions where YOU know the correct answer.
This lets you automatically test: did the RAG system get it right?

WHY THIS MATTERS:
On Day 2, hybrid search fixed 2 queries but broke 2 others.
You only caught the regressions because you manually checked every query.
At 100+ queries, manual checking is impossible.
A golden dataset automates this — run it after every change.

STRUCTURE OF EACH TEST CASE:
- question: what the user asks
- expected_answer: the correct answer (you write this)
- category: type of question (for analyzing patterns)
- expected_sources: which sections should be retrieved
- is_answerable: whether the answer exists in the docs
- difficulty: easy/medium/hard (for prioritization)

PM INSIGHT: Building the golden dataset is a PM job, not an engineering job.
You know the product, the users, and the edge cases better than engineers.
The quality of your eval is only as good as your test cases.
"""

# ──────────────────────────────────────────────
# THE GOLDEN DATASET
# ──────────────────────────────────────────────

GOLDEN_DATASET = [
    # ═══════════════════════════════════════════
    # CATEGORY: Personal Identity (simple lookups)
    # ═══════════════════════════════════════════
    {
        "id": "PI-001",
        "question": "What is Vishal's full name?",
        "expected_answer": "Vishal Goyal",
        "key_facts": ["Vishal Goyal"],
        "category": "personal_identity",
        "expected_sources": ["Section 1: Personal Identity"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "PI-002",
        "question": "What is Vishal's phone number?",
        "expected_answer": "+91 9611816024",
        "key_facts": ["+91 9611816024", "9611816024"],
        "category": "personal_identity",
        "expected_sources": ["Section 1: Personal Identity"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "PI-003",
        "question": "Where was Vishal born?",
        "expected_answer": "Singtam, East Sikkim, India",
        "key_facts": ["Singtam", "Sikkim"],
        "category": "personal_identity",
        "expected_sources": ["Section 1: Personal Identity"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "PI-004",
        "question": "Where does Vishal currently live?",
        "expected_answer": "Bengaluru, Karnataka, India",
        "key_facts": ["Bengaluru", "Bangalore", "Karnataka"],
        "category": "personal_identity",
        "expected_sources": ["Section 1: Personal Identity"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "PI-005",
        "question": "What is Vishal's email address?",
        "expected_answer": "vishallayog@gmail.com",
        "key_facts": ["vishallayog@gmail.com"],
        "category": "personal_identity",
        "expected_sources": ["Section 1: Personal Identity"],
        "is_answerable": True,
        "difficulty": "easy",
    },

    # ═══════════════════════════════════════════
    # CATEGORY: Education
    # ═══════════════════════════════════════════
    {
        "id": "ED-001",
        "question": "Where did Vishal do his graduation?",
        "expected_answer": "Presidency College, Bangalore. He pursued BCA (Bachelor of Computer Applications) from 2016-2019.",
        "key_facts": ["Presidency College", "BCA", "Bangalore"],
        "category": "education",
        "expected_sources": ["Section 2: Education Timeline"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "ED-002",
        "question": "What school did Vishal attend?",
        "expected_answer": "Holy Cross School in Singtam, Sikkim. He was there from 2004-2014. Before that, East Point School.",
        "key_facts": ["Holy Cross", "East Point", "Singtam"],
        "category": "education",
        "expected_sources": ["Section 2: Education Timeline"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "ED-003",
        "question": "What did Vishal study in 11th and 12th grade?",
        "expected_answer": "He studied at Jain College Jayanagar in Bangalore for his 11th and 12th grade.",
        "key_facts": ["Jain College", "Jayanagar", "Bangalore"],
        "category": "education",
        "expected_sources": ["Section 2: Education Timeline"],
        "is_answerable": True,
        "difficulty": "medium",
    },
    {
        "id": "ED-004",
        "question": "What product management training did Vishal do?",
        "expected_answer": "Product Management Fellowship at Upraised (Jul-Nov 2023). Also completed LinkedIn Learning certifications in Generative AI for Product Managers and MongoDB Aggregation Pipeline.",
        "key_facts": ["Upraised", "Product Management Fellowship", "Generative AI"],
        "category": "education",
        "expected_sources": ["Section 2: Education Timeline"],
        "is_answerable": True,
        "difficulty": "medium",
    },

    # ═══════════════════════════════════════════
    # CATEGORY: Career
    # ═══════════════════════════════════════════
    {
        "id": "CA-001",
        "question": "What was Vishal's first job?",
        "expected_answer": "Associate Analyst at Deloitte Consulting USI from August 2019 to September 2021. The client was HP.",
        "key_facts": ["Deloitte", "Associate Analyst", "2019", "HP"],
        "category": "career",
        "expected_sources": ["Section 3: Career Timeline"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "CA-002",
        "question": "What did Vishal achieve at HireQuotient?",
        "expected_answer": "Boosted engagement 4x (30 to 120 mins), achieved 95% AI accuracy, drove $1.2M ARR, introduced Hiring Manager collaboration module adding $50K revenue, reduced funnel drop-offs by 90%, improved outreach accuracy from 20% to 77%.",
        "key_facts": ["4x", "95%", "$1.2M", "ARR", "90%", "20%", "77%"],
        "category": "career",
        "expected_sources": ["Section 3: Career Timeline"],
        "is_answerable": True,
        "difficulty": "medium",
    },
    {
        "id": "CA-003",
        "question": "What is Vishal's current role?",
        "expected_answer": "Product Lead at Tech Mahindra, working as Senior Inbound Product Manager at ServiceNow since September 2025.",
        "key_facts": ["Tech Mahindra", "Product Lead", "ServiceNow", "2025"],
        "category": "career",
        "expected_sources": ["Section 3: Career Timeline"],
        "is_answerable": True,
        "difficulty": "hard",
    },
    {
        "id": "CA-004",
        "question": "How did Vishal transition from engineering to product management?",
        "expected_answer": "He quit his software engineering job in 2023, spent 4 months learning PM and networking, got an internship at HireQuotient, and converted it to full-time. He became PM within 6 months without an MBA or engineering degree.",
        "key_facts": ["quit", "4 months", "internship", "HireQuotient", "without MBA"],
        "category": "career",
        "expected_sources": ["Section 3: Career Timeline", "Section 7: Defining Moments"],
        "is_answerable": True,
        "difficulty": "medium",
    },
    {
        "id": "CA-005",
        "question": "What AI/ML work has Vishal done?",
        "expected_answer": "At HireQuotient: achieved 95% accuracy refining OpenAI prompts, designed GenAI-powered Copilot, optimized context for LLM to reduce costs. At ServiceNow: building agentic AI workflows, Task Mining automation with AI-driven solutions.",
        "key_facts": ["95%", "OpenAI", "prompt", "Copilot", "agentic", "Task Mining"],
        "category": "career",
        "expected_sources": ["Section 3: Career Timeline"],
        "is_answerable": True,
        "difficulty": "hard",
    },
    {
        "id": "CA-006",
        "question": "What companies has Vishal worked at?",
        "expected_answer": "Deloitte, Inviz AI Solutions, Labra.io, HireQuotient, and Tech Mahindra/ServiceNow.",
        "key_facts": ["Deloitte", "Inviz", "Labra", "HireQuotient", "Tech Mahindra"],
        "category": "career",
        "expected_sources": ["Section 3: Career Timeline"],
        "is_answerable": True,
        "difficulty": "medium",
    },

    # ═══════════════════════════════════════════
    # CATEGORY: Relationships
    # ═══════════════════════════════════════════
    {
        "id": "RE-001",
        "question": "Who is Sejal?",
        "expected_answer": "Vishal's girlfriend and college sweetheart from Presidency College. They have been together since college, in a long-distance relationship for 7+ years. She pursued MBA from IMI Delhi.",
        "key_facts": ["girlfriend", "Presidency", "long-distance", "7 years", "IMI Delhi"],
        "category": "relationships",
        "expected_sources": ["Section 4: Key Relationships"],
        "is_answerable": True,
        "difficulty": "medium",
    },
    {
        "id": "RE-002",
        "question": "Who is Ashish?",
        "expected_answer": "Best friend from Holy Cross School since 1st grade. Captain of Yellow House. Still one of Vishal's closest friends.",
        "key_facts": ["Holy Cross", "1st grade", "Yellow House", "closest friend"],
        "category": "relationships",
        "expected_sources": ["Section 4: Key Relationships"],
        "is_answerable": True,
        "difficulty": "medium",
    },
    {
        "id": "RE-003",
        "question": "Who are Vishal's college friends?",
        "expected_answer": "Kunal, Shikhar, Umang, and Sahil from Presidency College. Avijeet from school also joined Presidency.",
        "key_facts": ["Kunal", "Shikhar", "Umang", "Sahil", "Avijeet"],
        "category": "relationships",
        "expected_sources": ["Section 4: Key Relationships"],
        "is_answerable": True,
        "difficulty": "medium",
    },

    # ═══════════════════════════════════════════
    # CATEGORY: Interests & Skills
    # ═══════════════════════════════════════════
    {
        "id": "IN-001",
        "question": "What sports does Vishal follow?",
        "expected_answer": "Cricket and football. His favorite football team is Chelsea Football Club.",
        "key_facts": ["cricket", "football", "Chelsea"],
        "category": "interests",
        "expected_sources": ["Section 5: Interests"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "IN-002",
        "question": "What anime has Vishal watched?",
        "expected_answer": "Naruto, Bleach, and Haikyuu.",
        "key_facts": ["Naruto", "Bleach", "Haikyuu"],
        "category": "interests",
        "expected_sources": ["Section 5: Interests"],
        "is_answerable": True,
        "difficulty": "easy",
    },
    {
        "id": "IN-003",
        "question": "What programming languages does Vishal know?",
        "expected_answer": "Python. Also experienced with APIs, JSON, and tools like Algolia, Azure OAuth2.",
        "key_facts": ["Python"],
        "category": "skills",
        "expected_sources": ["Section 6: Technical Skills"],
        "is_answerable": True,
        "difficulty": "medium",
    },
    {
        "id": "IN-004",
        "question": "What product tools does Vishal use?",
        "expected_answer": "Jira, Figma, Whimsical, Postman, Intercom, Monday, Confluence, Mixpanel, Slack, ChatGPT, Claude, Claude Code, and others.",
        "key_facts": ["Jira", "Figma", "Postman", "Mixpanel", "Claude"],
        "category": "skills",
        "expected_sources": ["Section 6: Technical Skills"],
        "is_answerable": True,
        "difficulty": "medium",
    },

    # ═══════════════════════════════════════════
    # CATEGORY: Unanswerable (CRITICAL for hallucination testing)
    # ═══════════════════════════════════════════
    {
        "id": "UN-001",
        "question": "What is Vishal's salary?",
        "expected_answer": "NOT_ANSWERABLE",
        "key_facts": [],
        "category": "unanswerable",
        "expected_sources": [],
        "is_answerable": False,
        "difficulty": "easy",
    },
    {
        "id": "UN-002",
        "question": "What is Vishal's blood type?",
        "expected_answer": "NOT_ANSWERABLE",
        "key_facts": [],
        "category": "unanswerable",
        "expected_sources": [],
        "is_answerable": False,
        "difficulty": "easy",
    },
    {
        "id": "UN-003",
        "question": "What university did Sejal attend for her undergraduate degree?",
        "expected_answer": "NOT_ANSWERABLE",
        "key_facts": [],
        "category": "unanswerable",
        "expected_sources": [],
        "is_answerable": False,
        "difficulty": "medium",
    },
    {
        "id": "UN-004",
        "question": "How many people report to Vishal at Tech Mahindra?",
        "expected_answer": "NOT_ANSWERABLE",
        "key_facts": [],
        "category": "unanswerable",
        "expected_sources": [],
        "is_answerable": False,
        "difficulty": "medium",
    },
    {
        "id": "UN-005",
        "question": "What was Vishal's GPA in college?",
        "expected_answer": "NOT_ANSWERABLE",
        "key_facts": [],
        "category": "unanswerable",
        "expected_sources": [],
        "is_answerable": False,
        "difficulty": "easy",
    },

    # ═══════════════════════════════════════════
    # CATEGORY: Cross-cutting (answers span multiple sections)
    # ═══════════════════════════════════════════
    {
        "id": "CC-001",
        "question": "What are Vishal's leadership experiences?",
        "expected_answer": "Class monitor from 1st grade, football captain in 5th grade, popular student at Presidency, trained 4 engineers at Deloitte, led cross-functional teams at ServiceNow.",
        "key_facts": ["monitor", "captain", "trained", "cross-functional"],
        "category": "cross_cutting",
        "expected_sources": ["Section 2: Education Timeline", "Section 3: Career Timeline", "Section 7: Defining Moments"],
        "is_answerable": True,
        "difficulty": "hard",
    },
    {
        "id": "CC-002",
        "question": "Tell me about Vishal's journey from Sikkim to becoming a PM.",
        "expected_answer": "Born in Singtam Sikkim, studied at Holy Cross, moved to Bangalore for 11th grade at Jain College, BCA at Presidency, analyst at Deloitte, self-taught programming, software engineer, quit to pursue PM, fellowship at Upraised, internship at HireQuotient, became PM.",
        "key_facts": ["Singtam", "Bangalore", "Deloitte", "HireQuotient", "PM"],
        "category": "cross_cutting",
        "expected_sources": ["Section 2: Education Timeline", "Section 3: Career Timeline"],
        "is_answerable": True,
        "difficulty": "hard",
    },
    {
        "id": "CC-003",
        "question": "What failures or setbacks has Vishal faced?",
        "expected_answer": "Poor 4th grade results, poor 11th-12th results from over-enjoying, 9th grade relationship breakup, initial low CRM adoption at HireQuotient, outreach accuracy starting at 20%.",
        "key_facts": ["4th grade", "11th", "breakup", "CRM adoption", "20%"],
        "category": "cross_cutting",
        "expected_sources": ["Section 2: Education Timeline", "Section 3: Career Timeline", "Section 7: Defining Moments"],
        "is_answerable": True,
        "difficulty": "hard",
    },
]


def get_dataset():
    """Return the full golden dataset."""
    return GOLDEN_DATASET


def get_by_category(category: str):
    """Filter dataset by category."""
    return [q for q in GOLDEN_DATASET if q["category"] == category]


def get_by_difficulty(difficulty: str):
    """Filter by difficulty."""
    return [q for q in GOLDEN_DATASET if q["difficulty"] == difficulty]


def print_dataset_stats():
    """Print summary stats of the golden dataset."""
    from collections import Counter
    
    categories = Counter(q["category"] for q in GOLDEN_DATASET)
    difficulties = Counter(q["difficulty"] for q in GOLDEN_DATASET)
    answerable = sum(1 for q in GOLDEN_DATASET if q["is_answerable"])
    
    print(f"Total test cases: {len(GOLDEN_DATASET)}")
    print(f"Answerable: {answerable} | Unanswerable: {len(GOLDEN_DATASET) - answerable}")
    print(f"\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")


if __name__ == "__main__":
    print_dataset_stats()
