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

def gen_invoice():
    company = rand_company()
    inv_num = random.randint(10000, 99999)
    templates = [
        f"INVOICE #{inv_num}\n\n{company}\nDate: {rand_date()}\nDue Date: {rand_date()}\n\n"
        f"Bill To: {rand_company()}\n\n"
        f"Description: Professional services rendered\nQuantity: {random.randint(1,40)}\n"
        f"Rate: {rand_amount()}\nSubtotal: {rand_amount()}\nTax: {rand_amount()}\n"
        f"TOTAL DUE: {rand_amount()}\n\nPayment Terms: Net 30. Please remit payment to the "
        f"account listed below.",

        f"{company} — INVOICE\n\nInvoice Number: INV-{inv_num}\nIssue Date: {rand_date()}\n\n"
        f"Item: Consulting hours ({random.randint(5,80)} hrs @ {rand_amount()}/hr)\n"
        f"Amount Due: {rand_amount()}\n\nLate payments are subject to a 1.5% monthly fee. "
        f"Thank you for your business.",
    ]
    return random.choice(templates)


def gen_report():
    company = rand_company()
    templates = [
        f"QUARTERLY BUSINESS REVIEW — {company}\n\nExecutive Summary: Revenue grew to "
        f"{rand_amount()} this quarter, a {random.randint(2,40)}% increase year over year. "
        f"Key drivers included expansion into new markets and improved retention.\n\n"
        f"Operational Highlights: The team shipped {random.randint(3,20)} major releases. "
        f"Customer satisfaction scores rose to {random.randint(70,98)}%.\n\n"
        f"Outlook: Management expects continued growth into next quarter, with projected "
        f"revenue of {rand_amount()}.",

        f"ANNUAL PERFORMANCE REPORT\n\nPrepared for: {company} Board of Directors\n"
        f"Period: Fiscal Year {random.randint(2022,2026)}\n\n"
        f"Total revenue reached {rand_amount()}, compared to {rand_amount()} the prior year. "
        f"Operating expenses were {rand_amount()}. Net margin improved by {random.randint(1,15)} "
        f"percentage points. The board recommends continued investment in R&D.",
    ]
    return random.choice(templates)


def gen_policy():
    company = rand_company()
    templates = [
        f"{company} EMPLOYEE HANDBOOK — REMOTE WORK POLICY\n\nEffective {rand_date()}, all "
        f"employees are eligible to work remotely up to {random.randint(1,5)} days per week, "
        f"subject to manager approval. Employees must remain reachable during core hours "
        f"(10am-3pm local time). Equipment requests should be submitted to IT at least "
        f"{random.randint(3,14)} business days in advance.",

        f"DATA PRIVACY AND SECURITY POLICY\n\n{company} requires all employees to complete "
        f"security training annually. Confidential data must not be stored on personal devices. "
        f"Any suspected data breach must be reported to the Security team within "
        f"{random.randint(1,24)} hours of discovery. Violations of this policy may result in "
        f"disciplinary action up to termination.",
    ]
    return random.choice(templates)

def gen_memo():
    sender, dept = rand_name(), random.choice(DEPTS)
    templates = [
        f"MEMORANDUM\n\nTo: All Staff\nFrom: {sender}, {dept}\nDate: {rand_date()}\n"
        f"Re: Office Closure Notice\n\nPlease be advised that the office will be closed on "
        f"{rand_date()} for scheduled maintenance. All employees should plan to work remotely "
        f"that day. Contact {rand_name()} with any questions.",

        f"INTERNAL MEMO\n\nFrom: {sender}\nTo: {dept} Team\nDate: {rand_date()}\n\n"
        f"This is a reminder that the deadline for {random.choice(['budget submissions','performance reviews','project status updates'])} "
        f"is {rand_date()}. Please ensure all materials are submitted on time to avoid delays "
        f"in the next planning cycle.",
    ]
    return random.choice(templates)


def gen_other():
    templates = [
        f"Meeting notes from {rand_date()} sync. Attendees discussed roadmap priorities for "
        f"next sprint. Action items were assigned to {rand_name()} and {rand_name()}. "
        f"Next check-in scheduled for {rand_date()}.",

        f"FAQ: How do I reset my password?\n\nTo reset your password, navigate to account "
        f"settings and click 'Forgot Password'. You will receive an email with a reset link "
        f"valid for 24 hours. Contact support if you do not receive the email.",

        f"Product changelog v{random.randint(1,9)}.{random.randint(0,20)}.{random.randint(0,9)} "
        f"— Added dark mode support, fixed a bug causing crashes on startup, improved load "
        f"times by {random.randint(10,60)}%.",
    ]
    return random.choice(templates)

    GENERATORS = {
    "contract": gen_contract,
    "invoice": gen_invoice,
    "report": gen_report,
    "policy": gen_policy,
    "memo": gen_memo,
    "other": gen_other,
}


def generate_dataset(n_per_class: int = 150) -> list[dict]:
    examples = []
    for label, gen_fn in GENERATORS.items():
        for _ in range(n_per_class):
            examples.append({"text": gen_fn(), "label": label})
    random.shuffle(examples)
    return examples

def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    all_examples = generate_dataset(n_per_class=150)
    split = int(len(all_examples) * 0.85)
    train, val = all_examples[:split], all_examples[split:]

    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(out_dir / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(train)} train / {len(val)} val examples across {len(LABELS)} classes")
    print(f"Saved to {out_dir}/train.jsonl and {out_dir}/val.jsonl")


if __name__ == "__main__":
    main()
