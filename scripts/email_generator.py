#!/usr/bin/env python3
"""
Email content generator for testing classification systems.
This script provides functions to generate realistic email content across
multiple categories with varying styles, subjects, and content structures.
"""
import random
import json
import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Extended classification categories
EMAIL_CATEGORIES = {
    # Original categories
    "intro": {
        "name": "Introduction",
        "description": "Emails introducing people, products, or services",
        "weight": 0.05,
    },
    "promotion": {
        "name": "Promotion",
        "description": "Marketing and promotional emails",
        "weight": 0.08,
    },
    "meeting": {
        "name": "Meeting",
        "description": "Meeting invitations, confirmations, and follow-ups",
        "weight": 0.08,
    },
    "unknown": {
        "name": "Unknown",
        "description": "Emails that don't fit any category",
        "weight": 0.02,
    },
    
    # General categories
    "newsletter": {
        "name": "Newsletter",
        "description": "Regular newsletters and updates",
        "weight": 0.05,
    },
    "notification": {
        "name": "Notification",
        "description": "System notifications, alerts, and updates",
        "weight": 0.05,
    },
    "transaction": {
        "name": "Transaction",
        "description": "Purchase confirmations and receipts",
        "weight": 0.04,
    },
    "invitation": {
        "name": "Invitation",
        "description": "Event invitations, webinars, and conferences",
        "weight": 0.04,
    },
    "alert": {
        "name": "Alert",
        "description": "Critical alerts and time-sensitive notifications",
        "weight": 0.03,
    },
    "announcement": {
        "name": "Announcement",
        "description": "Company or product announcements",
        "weight": 0.03,
    },
    "receipt": {
        "name": "Receipt",
        "description": "Purchase receipts and confirmations",
        "weight": 0.03,
    },
    "request": {
        "name": "Request",
        "description": "Requests for information or action",
        "weight": 0.03,
    },
    "social": {
        "name": "Social",
        "description": "Social media notifications and updates",
        "weight": 0.02,
    },
    "updates": {
        "name": "Updates",
        "description": "Status updates and progress reports",
        "weight": 0.03,
    },
    "security": {
        "name": "Security",
        "description": "Security alerts, password resets, and warnings",
        "weight": 0.03,
    },
    "personal": {
        "name": "Personal",
        "description": "Personal communications and messages",
        "weight": 0.02,
    },
    "bill": {
        "name": "Bill",
        "description": "Bills, invoices, and payment requests",
        "weight": 0.03,
    },
    "travel": {
        "name": "Travel",
        "description": "Travel itineraries and confirmations",
        "weight": 0.02,
    },
    "shipping": {
        "name": "Shipping",
        "description": "Shipping confirmations and tracking updates",
        "weight": 0.02,
    },
    "survey": {
        "name": "Survey",
        "description": "Survey requests and feedback forms",
        "weight": 0.02,
    },
    
    # HR categories
    "job_application": {
        "name": "Job Application",
        "description": "Job applications and inquiries",
        "weight": 0.02,
    },
    "employment_offer": {
        "name": "Employment Offer",
        "description": "Job offers and employment details",
        "weight": 0.02,
    },
    "onboarding": {
        "name": "Onboarding",
        "description": "New employee onboarding information",
        "weight": 0.01,
    },
    "performance_review": {
        "name": "Performance Review",
        "description": "Performance reviews and feedback",
        "weight": 0.01,
    },
    "benefits": {
        "name": "Benefits",
        "description": "Employee benefits information",
        "weight": 0.01,
    },
    "payroll": {
        "name": "Payroll",
        "description": "Payroll notifications and updates",
        "weight": 0.01,
    },
    
    # Finance categories
    "invoice": {
        "name": "Invoice",
        "description": "Invoices and billing statements",
        "weight": 0.02,
    },
    "expense_report": {
        "name": "Expense Report",
        "description": "Expense reports and reimbursements",
        "weight": 0.01,
    },
    "budget": {
        "name": "Budget",
        "description": "Budget planning and updates",
        "weight": 0.01,
    },
    "payment_confirmation": {
        "name": "Payment Confirmation",
        "description": "Confirmations of payments received",
        "weight": 0.02,
    },
    
    # Legal categories
    "contract": {
        "name": "Contract",
        "description": "Contracts and legal agreements",
        "weight": 0.02,
    },
    "legal_notice": {
        "name": "Legal Notice",
        "description": "Legal notices and compliance information",
        "weight": 0.01,
    },
    "nda": {
        "name": "NDA",
        "description": "Non-disclosure agreements",
        "weight": 0.01,
    },
    
    # Support categories
    "support": {
        "name": "Support",
        "description": "Technical support, help desk, and troubleshooting",
        "weight": 0.04,
    },
    "maintenance_notification": {
        "name": "Maintenance Notification",
        "description": "System maintenance notifications",
        "weight": 0.02,
    },
    
    # Other categories maintained for backward compatibility
    "report": {
        "name": "Report",
        "description": "Status reports, analytics, and data summaries",
        "weight": 0.03,
    },
    "news": {
        "name": "News",
        "description": "News updates and announcements",
        "weight": 0.02,
    },
    "feedback": {
        "name": "Feedback",
        "description": "Feedback requests and responses",
        "weight": 0.02,
    },
    "inquiry": {
        "name": "Inquiry",
        "description": "General questions and inquiries",
        "weight": 0.02,
    }
}

# Data for random email generation
COMPANIES = [
    "Acme Inc", "TechCorp", "GlobalSoft", "DataSystems", "InnovateTech", 
    "PeakPerformance", "FutureTech", "OptimaSolutions", "Quantum Dynamics", 
    "Apex Solutions", "SkyNet Technologies", "Horizon Enterprises",
    "Pinnacle Systems", "Aurora Innovations", "Catalyst Corp", "Evergreen Solutions"
]

PEOPLE_FIRST_NAMES = [
    "Alex", "Taylor", "Jordan", "Morgan", "Casey", "Riley", "Jamie", "Quinn", 
    "Avery", "Pat", "Sam", "Chris", "Robin", "Dana", "Terry", "Jessie",
    "Cameron", "Skyler", "Ellis", "Emerson", "Finley", "Harper", "Rowan", "Sawyer"
]

PEOPLE_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
    "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee"
]

PROJECTS = [
    "Alpha", "Beta", "Phoenix", "Horizon", "Quantum", "Nexus", "Velocity", 
    "Fusion", "Zenith", "Catalyst", "Eclipse", "Aurora", "Spectrum", "Odyssey",
    "Titan", "Pinnacle", "Genesis", "Omega", "Matrix", "Vector", "Apex", "Nova"
]

PRODUCTS = [
    "Laptop", "Smartphone", "Headphones", "Software License", "Smart Watch", 
    "Camera", "Tablet", "Monitor", "Printer", "Keyboard", "Mouse", "Docking Station",
    "External Drive", "VR Headset", "Game Console", "Router", "Server", "Cloud Storage"
]

PRODUCT_CATEGORIES = [
    "electronics", "software", "office supplies", "furniture", "apparel", 
    "home goods", "fitness equipment", "kitchen appliances", "tools", 
    "books", "toys", "beauty products", "outdoor gear", "pet supplies"
]

SERVICES = [
    "Cloud Storage", "Email", "VPN", "Subscription", "Membership", "Account", 
    "Premium Plan", "Streaming Service", "Web Hosting", "Domain Registration",
    "Data Backup", "Project Management", "CRM", "Analytics", "Marketing Automation"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

ROLES = [
    "Developer", "Manager", "Designer", "Analyst", "Engineer", "Specialist", 
    "Coordinator", "Consultant", "Director", "Lead", "Architect", "Administrator",
    "Strategist", "Advisor", "Officer", "Head", "Chief", "VP", "Associate"
]

DEPARTMENTS = [
    "Engineering", "Marketing", "Sales", "Product", "Design", "Research", 
    "Finance", "Customer Support", "Human Resources", "Operations", "Legal",
    "Administration", "Development", "Quality Assurance", "Data Science"
]

ISSUES = [
    "bug", "error", "crash", "vulnerability", "outage", "performance issue", 
    "security breach", "memory leak", "data corruption", "connectivity issue",
    "timeout", "service disruption", "data loss", "configuration error"
]

SYSTEMS = [
    "database", "server", "network", "application", "website", "API", 
    "mobile app", "cloud infrastructure", "authentication system", 
    "storage system", "payment gateway", "email service", "backup system"
]

EVENTS = [
    "conference", "webinar", "workshop", "seminar", "training", "hackathon", 
    "meetup", "networking event", "product launch", "award ceremony", 
    "team building", "retreat", "summit", "symposium", "exhibition"
]

LOCATIONS = [
    "Conference Room A", "Meeting Room 3", "Office", "Headquarters", 
    "Branch Office", "Main Campus", "Innovation Center", "Training Facility",
    "Executive Suite", "Collaboration Space", "Auditorium", "Boardroom"
]

VIRTUAL_PLATFORMS = [
    "Zoom", "Microsoft Teams", "Google Meet", "Webex", "Slack", 
    "Discord", "Skype", "GoToMeeting", "BlueJeans", "Whereby"
]

SEASONS = [
    "Spring", "Summer", "Fall", "Winter", "Holiday", "Black Friday", 
    "Cyber Monday", "New Year", "Back to School", "Memorial Day",
    "Labor Day", "Valentine's Day", "Easter", "Halloween", "Thanksgiving"
]

METRICS = [
    "Revenue", "User Growth", "Conversion Rate", "Engagement", "Retention", 
    "Churn Rate", "Satisfaction Score", "Performance", "Click-through Rate",
    "Bounce Rate", "Average Order Value", "Cost per Acquisition", "ROI",
    "NPS Score", "Time on Site", "Page Views", "Unique Visitors"
]

def generate_person_name() -> str:
    """Generate a random person name"""
    return f"{random.choice(PEOPLE_FIRST_NAMES)} {random.choice(PEOPLE_LAST_NAMES)}"

def generate_date_future(min_days=1, max_days=30) -> str:
    """Generate a random future date"""
    days_ahead = random.randint(min_days, max_days)
    future_date = datetime.now() + timedelta(days=days_ahead)
    return future_date.strftime("%A, %B %d, %Y")

def generate_date_past(min_days=1, max_days=30) -> str:
    """Generate a random past date"""
    days_ago = random.randint(min_days, max_days)
    past_date = datetime.now() - timedelta(days=days_ago)
    return past_date.strftime("%A, %B %d, %Y")

def generate_time() -> str:
    """Generate a random time"""
    hour = random.randint(8, 17)
    minute = random.choice(["00", "15", "30", "45"])
    am_pm = "AM" if hour < 12 else "PM"
    hour = hour if hour <= 12 else hour - 12
    return f"{hour}:{minute} {am_pm}"

def generate_url() -> str:
    """Generate a random URL"""
    domains = ["example.com", "company.org", "service.net", "project.io", "team.dev", "product.app"]
    paths = ["dashboard", "account", "settings", "projects", "reports", "meetings", "documents", "files"]
    
    domain = random.choice(domains)
    path = random.choice(paths)
    
    if random.random() < 0.3:
        # Add a parameter
        param_names = ["id", "user", "ref", "source", "campaign", "token"]
        param_values = ["123456", "user123", "email", "newsletter", "summer2023", "abcdef"]
        param_name = random.choice(param_names)
        param_value = random.choice(param_values)
        return f"https://www.{domain}/{path}?{param_name}={param_value}"
    else:
        return f"https://www.{domain}/{path}"

def generate_percentage() -> int:
    """Generate a random percentage value"""
    # More likely to be "round" numbers
    if random.random() < 0.7:
        return random.choice([10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    else:
        return random.randint(5, 95)

def generate_reference_number() -> str:
    """Generate a random reference number"""
    if random.random() < 0.5:
        # Numeric
        return str(random.randint(10000, 999999))
    else:
        # Alphanumeric
        letters = ''.join(random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(2))
        numbers = ''.join(random.choice('0123456789') for _ in range(4))
        return f"{letters}{numbers}"

def generate_meeting_email() -> Dict[str, str]:
    """Generate a meeting invitation email"""
    # Pick a meeting type
    meeting_types = ["team meeting", "project update", "status review", "planning session", 
                    "brainstorming", "1:1", "kickoff", "retrospective", "demo"]
    
    meeting_type = random.choice(meeting_types)
    project = random.choice(PROJECTS)
    date = generate_date_future()
    time = generate_time()
    duration = random.choice(["30 minutes", "1 hour", "1.5 hours", "2 hours"])
    
    # Decide if virtual or in-person
    is_virtual = random.random() < 0.7  # 70% chance of virtual meeting
    
    if is_virtual:
        platform = random.choice(VIRTUAL_PLATFORMS)
        meeting_id = ''.join(random.choice('0123456789') for _ in range(9))
        password = ''.join(random.choice('abcdefghijkmnopqrstuvwxyz23456789') for _ in range(6))
        location = f"{platform} (Meeting ID: {meeting_id})"
        join_info = f"Join {platform}: https://{platform.lower()}.com/j/{meeting_id}\nPassword: {password}"
    else:
        location = random.choice(LOCATIONS)
        join_info = f"Location: {location}"
    
    # Create subject line
    subject_templates = [
        f"{meeting_type.title()}: {project} - {date}",
        f"Invitation: {meeting_type.title()} for {project} @ {time}",
        f"{project} {meeting_type.title()} - {date}, {time}",
        f"Please join: {meeting_type.title()} about {project}",
        f"{meeting_type.title()} regarding {project}"
    ]
    subject = random.choice(subject_templates)
    
    # Create email body
    sender_name = generate_person_name()
    
    # Generate attendees
    num_attendees = random.randint(1, 5)
    attendees = [generate_person_name() for _ in range(num_attendees)]
    attendees_str = ", ".join(attendees)
    
    # Generate agenda items
    num_agenda_items = random.randint(2, 5)
    agenda_templates = [
        "Review {item}",
        "Discuss {item}",
        "Update on {item}",
        "Plan for {item}",
        "Brainstorm {item} ideas",
        "Finalize {item}",
        "Analyze {item} results",
        "Approve {item} changes"
    ]
    
    agenda_topics = [
        "project timeline", "budget", "requirements", "design", "implementation", 
        "testing", "deployment", "marketing", "customer feedback", "next steps",
        "action items", "blockers", "priorities", "resources", "goals"
    ]
    
    agenda_items = []
    for _ in range(num_agenda_items):
        template = random.choice(agenda_templates)
        topic = random.choice(agenda_topics)
        agenda_items.append(template.format(item=topic))
    
    agenda_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(agenda_items)])
    
    # Build the email body
    body_templates = [
        f"""
Hi Team,

I'd like to invite you to a {meeting_type} for the {project} project.

Date: {date}
Time: {time} ({duration})
{join_info}

Agenda:
{agenda_str}

Please let me know if you have any questions or items to add to the agenda.

Regards,
{sender_name}
""",
        f"""
Hello everyone,

This is a calendar invitation for our upcoming {meeting_type}.

• Project: {project}
• Date: {date}
• Time: {time}
• Duration: {duration}
• {join_info}

Attendees: {attendees_str}

Agenda:
{agenda_str}

Looking forward to our discussion!

Best,
{sender_name}
""",
        f"""
Team,

Please join me for a {meeting_type} where we'll discuss the {project} project.

When: {date} at {time} ({duration})
Where: {location}

Topics to cover:
{agenda_str}

If you can't attend, please let me know as soon as possible.

Thanks,
{sender_name}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "meeting"
    }

def generate_intro_email() -> Dict[str, str]:
    """Generate an introduction email"""
    new_person = generate_person_name()
    sender = generate_person_name()
    role = random.choice(ROLES)
    department = random.choice(DEPARTMENTS)
    company = random.choice(COMPANIES)
    
    # Choose introduction context
    intro_contexts = ["new team member", "new client", "new partner", "new hire", "new manager", "new consultant"]
    context = random.choice(intro_contexts)
    
    # Create subject line
    subject_templates = [
        f"Introduction: {new_person}, our new {role}",
        f"Meet {new_person}, joining {department}",
        f"Welcome {new_person} to the team",
        f"Introducing {new_person} from {company}",
        f"New {context}: {new_person}"
    ]
    subject = random.choice(subject_templates)
    
    # Create background info
    background_templates = [
        f"{new_person} has {random.randint(2, 15)} years of experience in {random.choice(DEPARTMENTS).lower()}.",
        f"{new_person} comes to us from {random.choice(COMPANIES)}, where they worked as a {random.choice(ROLES).lower()}.",
        f"{new_person} has a background in {random.choice(DEPARTMENTS).lower()} and specializes in {random.choice(PROJECTS).lower()} projects.",
        f"{new_person} previously led the {random.choice(DEPARTMENTS).lower()} team at {random.choice(COMPANIES)}.",
        f"{new_person} brings valuable experience from their time at {random.choice(COMPANIES)}."
    ]
    background = random.choice(background_templates)
    
    # Create role info
    role_templates = [
        f"They will be joining our {department} team as a {role}.",
        f"In their role as {role}, they will be focusing on {random.choice(PROJECTS)}.",
        f"As our new {role}, they will help with {random.choice(PROJECTS)} and other initiatives.",
        f"They're joining the {department} department to work on {random.choice(PROJECTS)}.",
        f"Their primary focus as {role} will be to help us improve our {random.choice(SYSTEMS)}."
    ]
    role_info = random.choice(role_templates)
    
    # Build the email body
    body_templates = [
        f"""
Hello everyone,

I'd like to introduce {new_person}, who is joining us as a {role} in the {department} department.

{background}

{role_info}

Please join me in welcoming {new_person.split()[0]} to the team!

Best regards,
{sender}
""",
        f"""
Team,

I'm excited to introduce {new_person}, our new {role}.

{background}

{role_info}

{new_person.split()[0]}'s first day will be {generate_date_future(1, 10)}. Please stop by and introduce yourself!

Warm regards,
{sender}
""",
        f"""
Hi all,

Please welcome {new_person} to our {department} team!

About {new_person.split()[0]}:
• {background}
• {role_info}
• Email: {new_person.split()[0].lower()}@{company.lower().replace(' ', '')}.com

I'm looking forward to the contributions {new_person.split()[0]} will make to our team.

Cheers,
{sender}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "intro"
    }

def generate_promotion_email() -> Dict[str, str]:
    """Generate a promotional email"""
    product = random.choice(PRODUCTS)
    company = random.choice(COMPANIES)
    percentage = generate_percentage()
    season = random.choice(SEASONS)
    product_category = random.choice(PRODUCT_CATEGORIES)
    date = generate_date_future(3, 14)
    
    # Create subject line
    subject_templates = [
        f"{percentage}% Off {product} - {season} Sale!",
        f"SALE: Save {percentage}% on all {product_category}",
        f"Special Offer: {product} at {percentage}% discount",
        f"Flash Sale! {percentage}% Off Everything",
        f"{season} Deals: Up to {percentage}% Off {product_category}"
    ]
    subject = random.choice(subject_templates)
    
    # Create promotional content
    highlight_templates = [
        f"Save BIG with {percentage}% off all {product_category}!",
        f"For a limited time only: {percentage}% discount on {product}!",
        f"Our biggest {season} sale ever! {percentage}% off!",
        f"Don't miss these amazing deals - up to {percentage}% off!",
        f"FLASH SALE: {percentage}% off {product_category} this week only!"
    ]
    highlight = random.choice(highlight_templates)
    
    # Create call to action
    cta_templates = [
        f"Shop now at {generate_url()}",
        f"Visit our website to claim your discount: {generate_url()}",
        f"Use code SAVE{percentage} at checkout",
        f"Click here to browse all deals: {generate_url()}",
        f"Order today to take advantage of this limited-time offer!"
    ]
    cta = random.choice(cta_templates)
    
    # Build the email body
    body_templates = [
        f"""
{highlight}

Our {season} Sale is ON!

Take advantage of these incredible savings:
• {percentage}% off {product}
• Free shipping on orders over ${random.randint(25, 100)}
• Buy one, get one 50% off on select {product_category}

Sale ends {date}, so hurry!

{cta}

The {company} Team
""",
        f"""
SPECIAL OFFER

{highlight}

Why shop with us?
✓ {percentage}% discount on all {product_category}
✓ Free returns within 30 days
✓ 24/7 customer support
✓ Fast shipping

This offer expires on {date}!

{cta}

Thank you for being a valued customer!
{company}
""",
        f"""
🔥 HOT DEAL ALERT! 🔥

{highlight}

{percentage}% OFF
ALL {product_category.upper()}!

Including our best-selling {product}!

LIMITED TIME OFFER - Sale ends {date}

{cta}

Unsubscribe: {generate_url()}
{company} - Bringing you the best deals since {random.randint(1990, 2020)}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "promotion"
    }

def generate_report_email() -> Dict[str, str]:
    """Generate a report or status update email"""
    project = random.choice(PROJECTS)
    department = random.choice(DEPARTMENTS)
    sender = generate_person_name()
    
    # Time period
    periods = ["weekly", "monthly", "quarterly", "yearly", "sprint", "Q1", "Q2", "Q3", "Q4", "mid-year", "year-end"]
    period = random.choice(periods)
    
    # Select metrics
    selected_metrics = random.sample(METRICS, k=random.randint(3, 5))
    metrics_data = {}
    
    for metric in selected_metrics:
        if "Rate" in metric or "Percentage" in metric:
            metrics_data[metric] = f"{random.randint(1, 99)}%"
        elif "Score" in metric:
            metrics_data[metric] = f"{random.randint(1, 10)}/10" if random.random() < 0.5 else f"{random.randint(1, 100)}"
        else:
            metrics_data[metric] = f"{random.randint(100, 100000):,}"
    
    # Create subject line
    subject_templates = [
        f"{period.title()} Report: {project} Performance",
        f"{project} {period.title()} Status Update",
        f"{period.title()} {project} Metrics",
        f"{department} {period.title()} Report",
        f"{project} Performance: {period.title()} Summary"
    ]
    subject = random.choice(subject_templates)
    
    # Generate summary assessment
    assessment_templates = [
        f"Overall, we're {random.choice(['on track', 'making good progress', 'exceeding expectations', 'slightly behind schedule but catching up'])}.",
        f"The {project} project is {random.choice(['performing well', 'showing strong results', 'meeting our targets', 'facing some challenges'])}.",
        f"This {period}'s results {random.choice(['exceed our expectations', 'are in line with projections', 'show areas for improvement', 'demonstrate steady growth'])}.",
        f"We've seen {random.choice(['significant improvement', 'steady progress', 'some challenges', 'mixed results'])} in this {period}.",
        f"The team has {random.choice(['accomplished a lot', 'made substantial progress', 'overcome several obstacles', 'identified key areas for improvement'])}."
    ]
    assessment = random.choice(assessment_templates)
    
    # Generate next steps
    next_steps_templates = [
        f"Our focus for next {period} will be on {random.choice(['improving', 'optimizing', 'enhancing', 'reviewing'])} our {random.choice(METRICS).lower()}.",
        f"We plan to {random.choice(['roll out', 'implement', 'develop', 'finalize'])} the new {random.choice(SYSTEMS)} in the coming weeks.",
        f"The team will be {random.choice(['working on', 'addressing', 'tackling', 'prioritizing'])} the {random.choice(ISSUES)} in our {random.choice(SYSTEMS)}.",
        f"Next steps include {random.choice(['scheduling a review', 'updating our roadmap', 'conducting user testing', 'gathering more data'])}.",
        f"We'll be {random.choice(['meeting', 'following up', 'connecting', 'collaborating'])} to discuss these results in detail."
    ]
    next_steps = random.choice(next_steps_templates)
    
    # Create metrics summary
    metrics_summary = "\n".join([f"• {metric}: {value}" for metric, value in metrics_data.items()])
    
    # Build the email body
    body_templates = [
        f"""
Hi Team,

Here is the {period} report for the {project} project.

Key Metrics:
{metrics_summary}

{assessment}

{next_steps}

Please let me know if you have any questions or need additional information.

Best regards,
{sender}
""",
        f"""
Team,

Please find below the {period} performance summary for {project}.

SUMMARY:
{assessment}

METRICS:
{metrics_summary}

NEXT STEPS:
{next_steps}

The complete report is available at: {generate_url()}

Thank you,
{sender}
{department}
""",
        f"""
Hello everyone,

I'm sharing our {period} {project} metrics for your review.

{assessment}

Performance Highlights:
{metrics_summary}

Looking Ahead:
{next_steps}

I'm available to discuss these results in more detail if needed.

Regards,
{sender}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "report"
    }

def generate_support_email() -> Dict[str, str]:
    """Generate a technical support email"""
    issue = random.choice(ISSUES)
    system = random.choice(SYSTEMS)
    sender = generate_person_name()
    reference = generate_reference_number()
    
    # Create subject line
    subject_templates = [
        f"Technical Support: {issue} in {system}",
        f"Support Ticket #{reference}: {issue}",
        f"Help with {system} {issue}",
        f"{system} troubleshooting request",
        f"Technical assistance needed: {issue}"
    ]
    subject = random.choice(subject_templates)
    
    # Create issue description
    issue_description_templates = [
        f"I'm experiencing a {issue} with our {system}. It started {random.choice(['yesterday', 'this morning', 'a few hours ago', 'last week'])}.",
        f"Our {system} is showing a {issue}. This is affecting {random.choice(['a few users', 'our entire team', 'critical operations', 'our production environment'])}.",
        f"We're seeing an unusual {issue} in the {system}. It occurs when we try to {random.choice(['log in', 'save data', 'generate reports', 'process transactions', 'update records'])}.",
        f"There appears to be a {issue} in our {system}. The error message says: '{random.choice(['Connection refused', 'Internal server error', 'Access denied', 'Resource not found', 'Timeout exceeded'])}'.",
        f"I need help resolving a {issue} with the {system}. We've tried {random.choice(['restarting', 'updating', 'reinstalling', 'reconfiguring'])} but the problem persists."
    ]
    issue_description = random.choice(issue_description_templates)
    
    # Create impact description
    impact_templates = [
        f"This issue is preventing our team from {random.choice(['completing their work', 'accessing critical data', 'serving customers', 'meeting deadlines'])}.",
        f"Because of this problem, we can't {random.choice(['process new orders', 'update customer information', 'generate reports', 'access our data'])}.",
        f"The impact is {random.choice(['significant', 'minimal', 'growing', 'limited to specific users'])} at this point.",
        f"This is affecting approximately {random.randint(1, 100)} users and is considered {random.choice(['critical', 'high priority', 'important', 'urgent'])}.",
        f"We need this resolved {random.choice(['as soon as possible', 'within 24 hours', 'by the end of the week', 'urgently'])} as it's impacting {random.choice(['revenue', 'customer satisfaction', 'productivity', 'our ability to deliver'])}."
    ]
    impact = random.choice(impact_templates)
    
    # Build the email body
    body_templates = [
        f"""
Hello Support Team,

{issue_description}

{impact}

Here are some additional details:
• System: {system}
• Error occurs: {random.choice(['consistently', 'intermittently', 'only during peak hours', 'after system updates'])}
• First noticed: {generate_date_past(1, 7)}
• Error message: "{random.choice(['Error code 500', 'Connection timeout', 'Access denied', 'Data corruption detected', 'Service unavailable'])}"

Please let me know if you need any more information to help resolve this issue.

Thanks,
{sender}
""",
        f"""
Support Request

Ticket #: {reference}
Priority: {random.choice(['High', 'Medium', 'Low', 'Critical'])}
System: {system}
Issue: {issue}

Description:
{issue_description}

Impact:
{impact}

What I've tried:
• {random.choice(['Refreshed the page', 'Restarted the application', 'Cleared cache', 'Updated to latest version'])}
• {random.choice(['Checked system requirements', 'Verified network connectivity', 'Reviewed logs', 'Tested in different environment'])}
• {random.choice(['Contacted my IT department', 'Searched knowledge base', 'Asked colleagues if they experience the same', 'Checked for recent changes'])}

Thank you for your assistance,
{sender}
""",
        f"""
Hi Support,

We need technical assistance with our {system}.

Problem: {issue}

Details:
{issue_description}

{impact}

Screenshots and logs are attached. Please advise on next steps or let me know if you need remote access to diagnose the issue.

Best,
{sender}
Contact: {sender.lower().replace(' ', '.')}@{random.choice(['company.com', 'example.org', 'business.net'])}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "support"
    }

def generate_security_email() -> Dict[str, str]:
    """Generate a security alert or notification email"""
    service = random.choice(SERVICES)
    system = random.choice(SYSTEMS)
    company = random.choice(COMPANIES)
    
    # Security event types
    security_events = [
        "password reset", "unusual login attempt", "security update", "2FA setup",
        "account verification", "privacy policy update", "security alert",
        "suspicious activity", "device verification", "security breach"
    ]
    
    event = random.choice(security_events)
    
    # Create subject line
    subject_templates = [
        f"Security Alert: {event.title()} Required",
        f"Important: {event.title()} for Your {service} Account",
        f"{company} Security: {event.title()}",
        f"Action Required: {event.title()}",
        f"Security Notification: {event.title()}"
    ]
    subject = random.choice(subject_templates)
    
    # Generate security message
    if "password reset" in event:
        security_message = f"We received a request to reset the password for your {service} account. If you didn't make this request, please ignore this email or contact support immediately."
        
    elif "unusual login" in event:
        security_message = f"We detected a login to your {service} account from an unrecognized {random.choice(['device', 'location', 'IP address', 'browser'])}. If this wasn't you, please secure your account immediately."
        
    elif "security update" in event:
        security_message = f"We've released important security updates for our {system}. We recommend updating as soon as possible to protect your data and account."
        
    elif "2FA" in event:
        security_message = f"To enhance the security of your {service} account, we now support two-factor authentication. This adds an extra layer of protection beyond just a password."
        
    elif "verification" in event:
        security_message = f"Please verify your identity to continue using your {service} account. This is a routine security measure to protect your account."
        
    elif "policy update" in event:
        security_message = f"We've updated our privacy and security policies. These changes will take effect on {generate_date_future(7, 30)}."
        
    elif "suspicious" in event:
        security_message = f"We've detected suspicious activity on your {service} account. As a precautionary measure, we've temporarily restricted some account features."
        
    elif "breach" in event:
        security_message = f"We recently discovered unauthorized access to certain {company} systems. While your account was not directly affected, we recommend changing your password as a precaution."
        
    else:
        security_message = f"This is an important security notification regarding your {service} account. Please review the details below and take appropriate action."
    
    # Generate action items
    action_templates = [
        f"Please click here to {random.choice(['reset your password', 'verify your account', 'update your security settings', 'review recent activity'])}: {generate_url()}",
        f"To secure your account, {random.choice(['change your password', 'enable 2FA', 'review connected devices', 'update your recovery email'])}: {generate_url()}",
        f"If you recognize this activity, no action is needed. If not, please {random.choice(['contact us immediately', 'secure your account', 'report the issue'])}: {generate_url()}",
        f"Follow these steps to {random.choice(['protect your account', 'verify your identity', 'update your settings', 'report unauthorized access'])}: {generate_url()}",
        f"For more information, please visit our {random.choice(['Help Center', 'Security FAQ', 'Support Page', 'Account Security Guide'])}: {generate_url()}"
    ]
    action = random.choice(action_templates)
    
    # Build the email body
    body_templates = [
        f"""
Dear Customer,

{security_message}

Details:
• Account: {random.choice(['username', 'email'])}: ***{random.choice(['user', 'account', 'member', 'client'])}***
• Date & Time: {generate_date_past(0, 2)} at {generate_time()}
• {random.choice(['IP Address', 'Location', 'Device'])}: {random.choice(['192.168.x.x', 'New York, USA', 'Windows PC', 'Android Device', 'Unknown'])}

{action}

If you didn't initiate this action, please contact our support team immediately.

Regards,
{company} Security Team
""",
        f"""
SECURITY NOTIFICATION

{security_message}

What happened:
An {event} was detected on {generate_date_past(0, 3)} at {generate_time()}.

What this means:
Your account security is very important to us. This notification is part of our ongoing efforts to keep your {service} account secure.

What you should do:
{action}

Need help?
If you have any questions, contact us at security@{company.lower().replace(' ', '')}.com

Thank you,
{company} Security
""",
        f"""
Important Security Alert

{security_message}

⚠️ Action may be required

Your security is our priority. To protect your account:

1. {random.choice(['Verify it was you', 'Reset your password', 'Review account activity', 'Update security settings'])}
2. {random.choice(['Enable two-factor authentication', 'Update recovery information', 'Review connected applications', 'Check for suspicious activity'])}
3. {random.choice(['Ensure your email is secure', 'Use a strong, unique password', 'Keep your software updated', 'Be wary of phishing attempts'])}

{action}

This email was sent to you as part of our security procedures. It cannot be replied to.

{company}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "security"
    }

def generate_billing_email() -> Dict[str, str]:
    """Generate a billing or invoice email"""
    company = random.choice(COMPANIES)
    service = random.choice(SERVICES)
    reference = generate_reference_number()
    product = random.choice(PRODUCTS)
    sender = generate_person_name()
    
    # Payment details
    amount = f"${random.randint(10, 999)}.{random.choice(['00', '99', '50'])}"
    due_date = generate_date_future(3, 14)
    invoice_date = generate_date_past(1, 7)
    payment_methods = ["Credit Card", "Bank Transfer", "PayPal", "Check", "Direct Debit"]
    payment_method = random.choice(payment_methods)
    
    # Create subject line
    subject_templates = [
        f"Invoice #{reference} for {service}",
        f"Your {company} bill is ready - ${amount}",
        f"Payment due: {service} subscription",
        f"{company} billing statement - {invoice_date}",
        f"Receipt for your recent purchase: {product}"
    ]
    subject = random.choice(subject_templates)
    
    body_templates = [
        f"""
Dear Customer,

This is to confirm that your invoice #{reference} has been generated for your {service} subscription.

Invoice details:
• Invoice date: {invoice_date}
• Amount due: {amount}
• Due date: {due_date}
• Payment method: {payment_method}

To view or download your invoice, please visit: {generate_url()}

If you have any questions about this invoice, please contact our billing department at billing@{company.lower().replace(' ', '')}.com.

Thank you for your business,
{sender}
{company} Billing Team
""",
        f"""
BILLING NOTIFICATION

Invoice #: {reference}
Date: {invoice_date}
Due: {due_date}
Amount: {amount}

This is an automated notification that your {service} payment is now due.

Payment Options:
1. Online: {generate_url()}
2. Phone: Call (555) 123-4567
3. Mail: Send check to {company}, P.O. Box 12345, Anytown, USA

Please ensure payment is made by the due date to avoid service interruption.

Regards,
{company} Finance Department
""",
        f"""
Receipt - {company}

Thank you for your payment of {amount} for {service}.

Transaction Details:
• Date: {invoice_date}
• Reference: {reference}
• Item: {service} {random.choice(['Monthly', 'Annual', 'Quarterly'])} Subscription
• Next billing date: {generate_date_future(28, 32)}

Your payment has been processed successfully. This email serves as your official receipt.

Questions about your bill? Visit our billing FAQ: {generate_url()}

{company} - We appreciate your business!
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "bill"
    }

def generate_invitation_email() -> Dict[str, str]:
    """Generate an invitation to an event"""
    event = random.choice(EVENTS)
    sender = generate_person_name()
    company = random.choice(COMPANIES)
    date = generate_date_future(7, 60)
    time = generate_time()
    
    # Create subject line
    subject_templates = [
        f"Invitation: {company} {event} - {date}",
        f"You're invited: {event} on {date}",
        f"Join us for {company}'s upcoming {event}",
        f"Save the date: {event} - {date}",
        f"Exclusive invitation: {event} hosted by {company}"
    ]
    subject = random.choice(subject_templates)
    
    # Event details
    is_virtual = random.random() < 0.7
    if is_virtual:
        location = f"Online via {random.choice(VIRTUAL_PLATFORMS)}"
    else:
        location = random.choice(LOCATIONS)
    
    # Build the email body
    body_templates = [
        f"""
Dear Colleague,

You are cordially invited to attend our upcoming {event}.

Event Details:
• Date: {date}
• Time: {time}
• Location: {location}
• Host: {company}

{random.choice([
    f"This exclusive event will feature discussions on the latest trends in {random.choice(DEPARTMENTS).lower()}.",
    f"Join industry leaders for an insightful conversation about {random.choice(PROJECTS).lower()}.",
    f"Don't miss this opportunity to network with professionals in {random.choice(DEPARTMENTS).lower()}."
])}

Please RSVP by {generate_date_future(1, 5)} at {generate_url()}

We look forward to your participation!

Best regards,
{sender}
Event Coordinator, {company}
""",
        f"""
YOU'RE INVITED!

{company} presents:
{event.upper()}

When: {date} at {time}
Where: {location}

What to expect:
• {random.choice(['Keynote presentations', 'Panel discussions', 'Networking opportunities', 'Product demonstrations'])}
• {random.choice(['Q&A sessions', 'Hands-on workshops', 'Industry insights', 'Exclusive previews'])}
• {random.choice(['Complimentary refreshments', 'Expert speakers', 'Certificate of attendance', 'Resource materials'])}

{random.choice(['Space is limited', 'Registration is required', 'Early bird discount available until ' + generate_date_future(1, 5)])}

Register now: {generate_url()}

We hope to see you there!
{sender}, {company}
""",
        f"""
SAVE THE DATE: {date.upper()}

{company} invites you to join us for our {event}.

Time: {time}
Location: {location}

This event is perfect for {random.choice(['industry professionals', 'team leaders', 'business executives', 'technical specialists'])} interested in {random.choice(['innovation', 'professional development', 'networking', 'industry trends'])}.

More information: {generate_url()}

RSVP: {generate_url()}

Looking forward to connecting with you!

Warm regards,
{sender}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "invitation"
    }

def generate_newsletter_email() -> Dict[str, str]:
    """Generate a newsletter email"""
    company = random.choice(COMPANIES)
    sender = generate_person_name()
    month = random.choice(MONTHS)
    topics = random.sample([
        "industry trends", "company updates", "new products", "feature highlights",
        "customer success stories", "upcoming events", "tips and tricks",
        "team spotlights", "market insights", "product roadmap"
    ], k=random.randint(3, 5))
    
    subject_templates = [
        f"{company} Newsletter - {month} Edition",
        f"{month} Updates from {company}",
        f"What's New at {company}: {month} Newsletter",
        f"{company} Insider: {month}'s Top Stories",
        f"Your {month} Briefing from {company}"
    ]
    subject = random.choice(subject_templates)
    
    # Create content snippets
    content_snippets = []
    for topic in topics:
        if "trend" in topic:
            content_snippets.append(f"**{topic.title()}**\n{random.choice(DEPARTMENTS)} is seeing a surge in {random.choice(PRODUCTS).lower()} adoption. Our analysis shows a {generate_percentage()}% increase in the past quarter.")
        elif "update" in topic:
            content_snippets.append(f"**{topic.title()}**\nExciting changes at {company}! We've {random.choice(['expanded our team', 'moved to a new office', 'launched a new initiative', 'restructured our departments'])}.")
        elif "product" in topic:
            content_snippets.append(f"**{topic.title()}**\nIntroducing our newest offering: {random.choice(PRODUCTS)}. Designed to {random.choice(['improve efficiency', 'reduce costs', 'enhance productivity', 'streamline operations'])}.")
        elif "success" in topic:
            content_snippets.append(f"**{topic.title()}**\nHow {random.choice(COMPANIES)} achieved a {generate_percentage()}% {random.choice(['increase in efficiency', 'reduction in costs', 'improvement in customer satisfaction'])} using our solutions.")
        elif "event" in topic:
            content_snippets.append(f"**{topic.title()}**\nJoin us at the upcoming {random.choice(EVENTS)} on {generate_date_future(14, 60)}. {random.choice(['Register early', 'Limited spots available', 'Special discount for subscribers'])}.")
        else:
            content_snippets.append(f"**{topic.title()}**\n{random.choice(['Check out our latest blog post', 'Watch our recent webinar', 'Download our new whitepaper'])} about {random.choice(PRODUCTS).lower()} at {generate_url()}")
    
    # Shuffle content snippets
    random.shuffle(content_snippets)
    content = "\n\n".join(content_snippets)
    
    body_templates = [
        f"""
{company} Newsletter - {month} Edition

Dear Subscriber,

Welcome to our {month} newsletter! Here's what's happening this month:

{content}

Want more updates? Follow us on social media or visit our website: {generate_url()}

To unsubscribe, click here: {generate_url()}

Best regards,
{sender}
{company} Marketing Team
""",
        f"""
{month}'S HIGHLIGHTS FROM {company.upper()}

Hello from all of us at {company}!

IN THIS ISSUE:
{content}

Read the full stories on our blog: {generate_url()}

We value your feedback! Let us know what topics you'd like to see in future newsletters.

Until next month,
{sender} and the {company} Team
""",
        f"""
{company} INSIDER

{month} Edition | Stay Informed, Stay Ahead

{content}

QUICK LINKS:
• {random.choice(['Subscribe to our podcast', 'Join our community', 'Check out our resources'])}
• {random.choice(['View past newsletters', 'Meet our team', 'Request a demo'])}
• {random.choice(['Contact support', 'Share feedback', 'Update preferences'])}

Copyright © {datetime.now().year} {company}. All rights reserved.
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "newsletter"
    }

def generate_notification_email() -> Dict[str, str]:
    """Generate a notification email"""
    company = random.choice(COMPANIES)
    system = random.choice(SYSTEMS)
    
    notification_types = [
        "account update", "system maintenance", "feature release", 
        "policy change", "status change", "reminder", "confirmation"
    ]
    notification_type = random.choice(notification_types)
    
    subject_templates = [
        f"Notification: {notification_type} for your {system}",
        f"Important: {notification_type.title()} - Action Required",
        f"{company} Alert: {notification_type.title()}",
        f"Your {system} {notification_type} notification",
        f"{notification_type.title()} - Please Review"
    ]
    subject = random.choice(subject_templates)
    
    # Create notification content based on type
    if "maintenance" in notification_type:
        content = f"We will be performing scheduled maintenance on our {system} on {generate_date_future(1, 7)} from {generate_time()} for approximately {random.choice(['1 hour', '2 hours', '30 minutes'])}. During this time, the service may be temporarily unavailable."
    elif "feature" in notification_type:
        content = f"We're excited to announce that we've released new features for {system}. These improvements include {random.choice(['enhanced performance', 'additional functionality', 'improved user interface', 'better security'])}."
    elif "policy" in notification_type:
        content = f"We've updated our {random.choice(['terms of service', 'privacy policy', 'user agreement', 'data retention policy'])} effective {generate_date_future(7, 30)}. Please review the changes at {generate_url()}"
    elif "reminder" in notification_type:
        content = f"This is a friendly reminder about your upcoming {random.choice(['subscription renewal', 'payment due date', 'account review', 'password expiration'])} on {generate_date_future(1, 7)}."
    elif "confirmation" in notification_type:
        content = f"This email confirms your recent {random.choice(['account changes', 'subscription update', 'profile modification', 'settings adjustment'])} made on {generate_date_past(0, 2)}."
    else:
        content = f"We're contacting you regarding an important update to your {system} account. Please review the details below and take any necessary action."
    
    body_templates = [
        f"""
Dear User,

{content}

{random.choice([
    f"If you have any questions, please contact our support team.",
    f"No action is required from you at this time.",
    f"Please visit {generate_url()} for more information.",
    f"To update your notification preferences, visit your account settings."
])}

Thank you,
{company} Support Team
""",
        f"""
NOTIFICATION: {notification_type.upper()}

{content}

WHAT THIS MEANS FOR YOU:
• {random.choice(['Your account remains active', 'No immediate action required', 'Service will continue uninterrupted', 'Your data remains secure'])}
• {random.choice(['You may notice brief service interruptions', 'New features will be available automatically', 'Updated terms apply to your account', 'We recommend reviewing your settings'])}

For more details: {generate_url()}

This is an automated notification. Please do not reply to this email.

{company}
""",
        f"""
Important Notification

Re: {notification_type.title()}

Hello,

{content}

{random.choice([
    f"To acknowledge this notification, please click here: {generate_url()}",
    f"For assistance, contact us at support@{company.lower().replace(' ', '')}.com",
    f"You can manage your notification settings at any time through your account dashboard."
])}

Best regards,
{company} Notifications
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "notification"
    }

def generate_transaction_email() -> Dict[str, str]:
    """Generate a transaction confirmation email"""
    company = random.choice(COMPANIES)
    product = random.choice(PRODUCTS)
    reference = generate_reference_number()
    
    # Transaction details
    amount = f"${random.randint(10, 999)}.{random.choice(['00', '99', '50'])}"
    transaction_date = generate_date_past(0, 2)
    payment_methods = ["Credit Card ****" + ''.join(random.choice('0123456789') for _ in range(4)), 
                      "PayPal", "Bank Transfer", "Digital Wallet", "Gift Card"]
    payment_method = random.choice(payment_methods)
    
    subject_templates = [
        f"Transaction Confirmation #{reference}",
        f"Your {company} purchase receipt",
        f"Order Confirmation: {product}",
        f"Receipt for your payment of {amount}",
        f"Thank you for your purchase from {company}"
    ]
    subject = random.choice(subject_templates)
    
    body_templates = [
        f"""
Dear Customer,

Thank you for your purchase! Your transaction has been completed successfully.

Transaction Details:
• Date: {transaction_date}
• Amount: {amount}
• Reference: #{reference}
• Payment Method: {payment_method}
• Item: {product}

{random.choice([
    f"Your digital download is available at: {generate_url()}",
    f"Your order will be delivered within {random.randint(2, 7)} business days.",
    f"Your subscription has been activated and is ready to use.",
    f"Your receipt has been attached to this email for your records."
])}

If you have any questions about this transaction, please contact our customer service team.

Thank you for choosing {company}!

Best regards,
{company} Customer Service
""",
        f"""
TRANSACTION CONFIRMATION

Your payment has been processed successfully.

PAYMENT SUMMARY:
Item: {product}
Amount: {amount}
Date: {transaction_date}
Transaction ID: {reference}
Method: {payment_method}

{random.choice([
    f"Track your shipment: {generate_url()}",
    f"View your purchase history: {generate_url()}",
    f"Download your item: {generate_url()}",
    f"Manage your subscription: {generate_url()}"
])}

Need help? Contact us at support@{company.lower().replace(' ', '')}.com

{company} - Thank you for your business!
""",
        f"""
Receipt from {company}

We've received your payment of {amount}.

Order Details:
• Product: {product}
• Order #: {reference}
• Date: {transaction_date}
• Payment: {payment_method}

{random.choice([
    f"Your purchase includes free support for 30 days.",
    f"Access your account to view complete order details.",
    f"Save this receipt for your records and warranty claims.",
    f"A survey about your purchase experience will be sent separately."
])}

Questions about your order? Visit our Help Center or reply to this email.

{company}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "transaction"
    }

def generate_employment_offer_email() -> Dict[str, str]:
    """Generate an employment offer email"""
    company = random.choice(COMPANIES)
    role = random.choice(ROLES)
    department = random.choice(DEPARTMENTS)
    recipient = generate_person_name()
    sender = generate_person_name()
    start_date = generate_date_future(14, 45)
    
    subject_templates = [
        f"Job Offer: {role} position at {company}",
        f"Official Employment Offer - {role}",
        f"{company} is pleased to offer you the {role} position",
        f"Your offer letter for the {role} position",
        f"Welcome to {company}: Your Employment Offer"
    ]
    subject = random.choice(subject_templates)
    
    salary = f"${random.randint(50, 150)},000"
    benefits = [
        "Health, dental, and vision insurance",
        "401(k) plan with company match",
        "Flexible work arrangements",
        "Professional development budget",
        "Paid time off",
        "Stock options",
        "Performance bonuses",
        "Relocation assistance"
    ]
    selected_benefits = random.sample(benefits, k=random.randint(3, 5))
    benefits_list = "\n".join([f"• {benefit}" for benefit in selected_benefits])
    
    body_templates = [
        f"""
Dear {recipient},

We are pleased to offer you the position of {role} in our {department} department at {company}. We were impressed with your background and believe you would be a valuable asset to our team.

Position Details:
• Title: {role}
• Department: {department}
• Start Date: {start_date}
• Salary: {salary} per year
• Status: Full-time

Benefits:
{benefits_list}

To accept this offer, please sign and return the attached offer letter by {generate_date_future(3, 7)}. If you have any questions or need clarification on any aspect of this offer, please don't hesitate to contact me.

We are excited about the possibility of you joining our team and look forward to your response.

Best regards,
{sender}
Human Resources
{company}
""",
        f"""
OFFICIAL EMPLOYMENT OFFER

{company} → {recipient}

Dear {recipient},

Congratulations! Following our recent interviews, we are delighted to formally offer you the position of {role} at {company}.

OFFER DETAILS:
• Position: {role}
• Team: {department}
• Compensation: {salary} annually
• Start date: {start_date}

YOUR BENEFITS PACKAGE INCLUDES:
{benefits_list}

Next steps:
1. Review the attached formal offer letter and employment agreement
2. Sign and return by {generate_date_future(3, 7)}
3. Complete the background check process
4. Our HR team will contact you to arrange onboarding

We're thrilled about the skills and perspective you'll bring to our team. If you have any questions, please contact me directly.

Looking forward to welcoming you aboard!

{sender}
Talent Acquisition Manager
{company}
""",
        f"""
Welcome to {company}!

Dear {recipient},

It is with great pleasure that I officially offer you the position of {role} within our {department} department at {company}.

Based on our discussions and your interview process, we believe you are an excellent fit for our team and company culture. 

Your employment terms include:
• Annual salary: {salary}
• Starting date: {start_date}
• Location: {random.choice(['Headquarters', 'Remote', 'Regional Office'])}

Benefits summary:
{benefits_list}

To indicate your acceptance, please sign the enclosed employment agreement and return it by {generate_date_future(3, 7)}.

We're excited to have you join us and contribute to our continued success.

Sincerely,
{sender}
{company}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "employment_offer"
    }

def generate_contract_email() -> Dict[str, str]:
    """Generate a contract-related email"""
    company = random.choice(COMPANIES)
    partner_company = random.choice(COMPANIES)
    sender = generate_person_name()
    recipient = generate_person_name()
    
    contract_types = [
        "Service Agreement", "Master Services Agreement", "Statement of Work",
        "Partnership Agreement", "Non-Disclosure Agreement", "Licensing Agreement",
        "Sales Contract", "Consulting Agreement", "Vendor Agreement"
    ]
    contract_type = random.choice(contract_types)
    
    subject_templates = [
        f"{contract_type} for your review",
        f"Contract: {company} and {partner_company}",
        f"Please review: {contract_type} draft",
        f"{contract_type} - Requires your signature",
        f"Important: {contract_type} for upcoming project"
    ]
    subject = random.choice(subject_templates)
    
    due_date = generate_date_future(3, 10)
    
    body_templates = [
        f"""
Dear {recipient},

I am sending the {contract_type} between {company} and {partner_company} for your review and signature.

This contract outlines the terms and conditions for our {random.choice(['upcoming collaboration', 'project engagement', 'service provision', 'partnership arrangement'])}.

Key points include:
• Term: {random.randint(1, 3)} year(s), commencing on {generate_date_future(7, 30)}
• {random.choice(['Payment terms: Net 30', 'Deliverables schedule included', 'Performance metrics defined', 'Renewal options detailed'])}
• {random.choice(['Confidentiality provisions', 'Intellectual property rights', 'Termination clauses', 'Liability limitations'])}

Please review the attached document and provide your signature by {due_date}.

If you have any questions or need clarification on any terms, please don't hesitate to contact me.

Best regards,
{sender}
{random.choice(['Legal Department', 'Business Development', 'Contracts Manager', 'Operations'])}
{company}
""",
        f"""
{contract_type.upper()} - ACTION REQUIRED

Hello {recipient},

Attached please find the {contract_type} between {company} and {partner_company} for your review and execution.

AGREEMENT SUMMARY:
• Type: {contract_type}
• Parties: {company} and {partner_company}
• Purpose: {random.choice(['Professional services', 'Product licensing', 'Joint venture', 'Software development'])}
• Effective date: Upon final signature
• Duration: {random.choice(['12 months', '24 months', '36 months', 'Until project completion'])}

Please review, sign, and return by {due_date}. You can use our electronic signature system at {generate_url()} or sign the PDF and email it back.

Let me know if you need any changes or have questions about the terms.

Regards,
{sender}
{company}
""",
        f"""
Contract for Review: {contract_type}

Dear {recipient},

I hope this email finds you well. I'm reaching out regarding the {contract_type} between our organizations.

As discussed, I've attached the contract document for your review and approval. This agreement covers:

1. {random.choice(['Scope of work', 'Service specifications', 'Deliverables', 'Project timeline'])}
2. {random.choice(['Financial terms', 'Payment schedule', 'Pricing details', 'Budget allocation'])}
3. {random.choice(['Legal obligations', 'Compliance requirements', 'Warranty information', 'Support terms'])}

We would appreciate your feedback and signature by {due_date} to ensure we can {random.choice(['begin work on schedule', 'maintain our timeline', 'proceed with the next steps', 'finalize our arrangement'])}.

Please feel free to suggest any modifications or ask questions about specific clauses.

Thank you for your attention to this matter.

Sincerely,
{sender}
{company}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "contract"
    }

def generate_survey_email() -> Dict[str, str]:
    """Generate a survey or feedback request email"""
    company = random.choice(COMPANIES)
    product = random.choice(PRODUCTS)
    service = random.choice(SERVICES)
    item = random.choice([product, service])
    
    subject_templates = [
        f"Your feedback on {item} is important to us",
        f"Quick survey: Help us improve {item}",
        f"{company} customer satisfaction survey",
        f"We value your opinion about {item}",
        f"5 minutes of your time? Share your thoughts on {item}"
    ]
    subject = random.choice(subject_templates)
    
    body_templates = [
        f"""
Dear Customer,

We hope you've been enjoying your experience with {item}. Your feedback is invaluable to us as we continue to improve our offerings.

We've prepared a short survey that should take only {random.choice(['3-5 minutes', '5 minutes', 'a few moments'])} to complete. Your responses will help us understand how we can better serve you and other customers.

Survey link: {generate_url()}

{random.choice([
    f"As a thank you for your time, you'll be entered into a drawing for a {random.choice(['$50 gift card', '$100 account credit', 'free month of service'])}.",
    f"Your feedback directly influences our product roadmap and service improvements.",
    f"We read every response and take your suggestions seriously."
])}

Thank you for helping us improve!

Best regards,
{company} Customer Experience Team
""",
        f"""
WE WANT YOUR FEEDBACK!

Your opinion matters to us at {company}.

You recently {random.choice(['purchased', 'used', 'subscribed to', 'interacted with'])} our {item}, and we'd love to know about your experience.

OUR QUICK SURVEY ASKS ABOUT:
• Your overall satisfaction
• Ease of use
• Features you value most
• Areas for improvement
• Likelihood to recommend

It will take just {random.choice(['3', '5', '2-3'])} minutes of your time:
→ {generate_url()}

Your feedback helps us create better products and services for you.

Thank you!
{company} Team
""",
        f"""
Share Your Thoughts - {item} Feedback

Hello from {company},

We noticed you've been using {item} for a while now, and we'd really appreciate your insights.

Our product team is actively working on improvements, and your real-world experience is exactly what we need to guide our decisions.

Please take our brief survey:
{generate_url()}

{random.choice([
    f"This survey is completely anonymous, so please feel free to be candid.",
    f"Your specific comments will be reviewed by our product managers.",
    f"We've designed this survey to be quick and focused, respecting your time."
])}

Thank you for being a valued customer and for helping shape the future of {item}.

Regards,
{company}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "survey"
    }

def generate_job_application_email() -> Dict[str, str]:
    """Generate a job application email"""
    role = random.choice(ROLES)
    department = random.choice(DEPARTMENTS)
    company = random.choice(COMPANIES)
    sender = generate_person_name()
    reference = 'JOB' + ''.join(random.choice('0123456789') for _ in range(5))
    
    subject_templates = [
        f"Application for {role} position",
        f"{role} job application - {sender}",
        f"Job application: {role} at {company} ({reference})",
        f"Interested in the {role} position",
        f"Applying for {department} {role} role"
    ]
    subject = random.choice(subject_templates)
    
    skills = [
        f"Experience with {random.choice(SYSTEMS)}",
        f"Knowledge of {random.choice(DEPARTMENTS)} processes",
        f"Proficient in {random.choice(['Python', 'Java', 'C++', 'SQL', 'JavaScript', 'Ruby'])}",
        f"{random.randint(3, 10)}+ years in {random.choice(DEPARTMENTS)}",
        f"Strong background in {random.choice(['project management', 'data analysis', 'customer service', 'software development', 'marketing', 'sales'])}"
    ]
    selected_skills = random.sample(skills, k=random.randint(2, 4))
    skills_list = "\n".join([f"• {skill}" for skill in selected_skills])
    
    body_templates = [
        f"""
Dear Hiring Manager,

I am writing to apply for the {role} position in the {department} department at {company}, as advertised on {random.choice(['your company website', 'LinkedIn', 'Indeed', 'Glassdoor'])}.

With my background in {random.choice(DEPARTMENTS).lower()}, I believe I would be a valuable addition to your team. My relevant skills and experience include:

{skills_list}

My resume is attached for your review, detailing my professional experience and accomplishments. I am particularly interested in this position because of {random.choice([f"your company's innovative approach to {random.choice(SYSTEMS).lower()}", f"the opportunity to work on {random.choice(PROJECTS)}", f"your reputation as a leader in the industry", f"my passion for {department.lower()} work"])}.

I would welcome the opportunity to discuss how my skills and experience could benefit {company}. I am available for an interview at your convenience.

Thank you for considering my application. I look forward to hearing from you.

Sincerely,
{sender}
{sender.split()[0].lower()}.{sender.split()[1].lower()}@email.com
{random.choice(['555-123-4567', '555-987-6543', '555-234-5678'])}
""",
        f"""
JOB APPLICATION: {role.upper()} POSITION

Dear {company} Recruiting Team,

I am excited to submit my application for the {role} position in your {department} department (Job Reference: {reference}).

ABOUT ME:
I am a {random.choice(['passionate', 'detail-oriented', 'results-driven', 'innovative'])} professional with a strong background in {random.choice(DEPARTMENTS).lower()}.

KEY QUALIFICATIONS:
{skills_list}

WHY {company.upper()}?
{random.choice([f"I've long admired your work in {random.choice(PRODUCTS)}", f"Your company values align perfectly with my professional goals", f"Your reputation for {random.choice(['innovation', 'excellence', 'work culture', 'professional development'])}", f"I'm passionate about the problems you're solving"])}

I've attached my resume, portfolio, and references for your review. I'm available for interviews {random.choice(['at your convenience', 'starting next week', 'via video call or in person'])}.

Thank you for your consideration. I look forward to the possibility of contributing to your team.

Best regards,
{sender}
{sender.split()[0].lower()}@email.com
""",
        f"""
Application for {role} position at {company}

Hello,

I hope this email finds you well. I'm reaching out to express my interest in the {role} position within your {department} department.

After reading the job description, I'm confident that my experience and skills make me a strong candidate for this role.

Professional highlights:
{skills_list}

What attracts me to {company} is {random.choice([f"your commitment to {random.choice(['quality', 'innovation', 'customer satisfaction', 'sustainability'])}", f"the opportunity to work on {random.choice(PROJECTS)}", f"your industry-leading position in {random.choice(DEPARTMENTS).lower()}", f"the collaborative culture I've heard about from current employees"])}.

I've attached my resume and cover letter that further detail my qualifications. I would appreciate the opportunity to discuss how I can contribute to your team.

Thank you for considering my application.

Regards,
{sender}
{random.choice(['LinkedIn Profile:', 'Portfolio:', 'References available upon request'])} {generate_url()}
"""
    ]
    
    body = random.choice(body_templates)
    
    return {
        "subject": subject,
        "body": body,
        "category": "job_application"
    }

def generate_email(category=None) -> Dict[str, str]:
    """Generate a random email with specified category or random category"""
    # If no category specified, pick one based on weights
    if not category:
        categories = list(EMAIL_CATEGORIES.keys())
        weights = [EMAIL_CATEGORIES[cat]["weight"] for cat in categories]
        category = random.choices(categories, weights=weights)[0]
    
    # Map category to function
    category_functions = {
        "meeting": generate_meeting_email,
        "intro": generate_intro_email,
        "promotion": generate_promotion_email,
        "report": generate_report_email,
        "support": generate_support_email,
        "security": generate_security_email,
        "bill": generate_billing_email,
        "invitation": generate_invitation_email,
        "newsletter": generate_newsletter_email,
        "notification": generate_notification_email,
        "transaction": generate_transaction_email,
        "employment_offer": generate_employment_offer_email,
        "contract": generate_contract_email,
        "survey": generate_survey_email,
        "job_application": generate_job_application_email
    }
    
    # Generate email based on category
    if category in category_functions:
        return category_functions[category]()
    else:
        # Default to a random implemented category
        implemented = list(category_functions.keys())
        return category_functions[random.choice(implemented)]()

def generate_email_batch(count=10, categories=None) -> List[Dict[str, str]]:
    """Generate a batch of random emails, optionally from specific categories"""
    batch = []
    
    for _ in range(count):
        if categories:
            category = random.choice(categories)
            email = generate_email(category)
        else:
            email = generate_email()
        
        batch.append(email)
    
    return batch

def save_email_batch(batch, filename="email_templates.json"):
    """Save a batch of emails to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(batch, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate realistic email templates for testing")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of emails to generate")
    parser.add_argument("-o", "--output", type=str, default="email_templates.json", help="Output JSON file")
    parser.add_argument("--categories", type=str, help="Comma-separated list of categories to generate")
    
    args = parser.parse_args()
    
    categories = args.categories.split(",") if args.categories else None
    
    print(f"Generating {args.count} email templates...")
    batch = generate_email_batch(args.count, categories)
    
    save_email_batch(batch, args.output)
    print(f"Email templates saved to {args.output}")
    
    # Print category distribution
    cat_count = {}
    for email in batch:
        cat = email["category"]
        cat_count[cat] = cat_count.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in cat_count.items():
        print(f"  {cat}: {count} ({count/args.count*100:.1f}%)")

if __name__ == "__main__":
    main()