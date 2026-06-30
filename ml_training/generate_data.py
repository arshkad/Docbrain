# ─── Per-class template banks ───────────────────────────────────────────────────
# Each template is filled with randomized entities to create varied but
# structurally-faithful examples of that document type.

def gen_contract():
    a, b = rand_company(), rand_company()
    templates = [
        f"SERVICE AGREEMENT\n\nThis Agreement is entered into as of {rand_date()} between "
        f"{a} (\"Service Provider\") and {b} (\"Client\"). Service Provider agrees to deliver "
        f"the services described in Exhibit A. The term of this Agreement shall commence on "
        f"{rand_date()} and continue until {rand_date()} unless terminated earlier in accordance "
        f"with Section 7. Client shall pay Service Provider {rand_amount()} per the payment "
        f"schedule in Exhibit B. Either party may terminate this Agreement upon 30 days written notice.",

        f"MASTER SERVICES AGREEMENT between {a} and {b}, effective {rand_date()}.\n\n"
        f"1. SCOPE OF WORK. Provider shall perform the services set forth herein.\n"
        f"2. TERM AND TERMINATION. This Agreement remains in effect until {rand_date()}.\n"
        f"3. CONFIDENTIALITY. Both parties agree to maintain confidentiality of proprietary "
        f"information for a period of two years following termination.\n"
        f"4. GOVERNING LAW. This Agreement shall be governed by the laws of the state of Delaware.",

        f"NON-DISCLOSURE AGREEMENT\n\nThis Non-Disclosure Agreement (\"NDA\") is made between "
        f"{a} and {b} as of {rand_date()}. The parties wish to explore a potential business "
        f"relationship and in connection with this opportunity, each party may disclose "
        f"confidential information to the other. The receiving party agrees not to disclose "
        f"any confidential information to third parties for a period of three years.",
    ]
    return random.choice(templates)