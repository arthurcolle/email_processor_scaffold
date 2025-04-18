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
    "intro": {
        "name": "Introduction",
        "description": "Emails introducing people, products, or services",
        "weight": 0.10,
    },
    "promotion": {
        "name": "Promotion",
        "description": "Marketing and promotional emails",
        "weight": 0.15,
    },
    "meeting": {
        "name": "Meeting",
        "description": "Meeting invitations, confirmations, and follow-ups",
        "weight": 0.15,
    },
    "report": {
        "name": "Report",
        "description": "Status reports, analytics, and data summaries",
        "weight": 0.10,
    },
    "news": {
        "name": "News",
        "description": "News updates, announcements, and newsletters",
        "weight": 0.08,
    },
    "support": {
        "name": "Support",
        "description": "Technical support, help desk, and troubleshooting",
        "weight": 0.07,
    },
    "billing": {
        "name": "Billing",
        "description": "Invoices, receipts, and payment notifications",
        "weight": 0.05,
    },
    "security": {
        "name": "Security",
        "description": "Security alerts, password resets, and warnings",
        "weight": 0.04,
    },
    "invitation": {
        "name": "Invitation",
        "description": "Event invitations, webinars, and conferences",
        "weight": 0.05,
    },
    "feedback": {
        "name": "Feedback",
        "description": "Feedback requests, surveys, and reviews",
        "weight": 0.06,
    },
    "notification": {
        "name": "Notification",
        "description": "System notifications, alerts, and updates",
        "weight": 0.07,
    },
    "inquiry": {
        "name": "Inquiry",
        "description": "General questions, inquiries, and information requests",
        "weight": 0.08,
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

def generate_email(category=None) -> Dict[str, str]:
    """Generate a random email with specified category or random category"""
    # If no category specified, pick one based on weights
    if not category:
        categories = list(EMAIL_CATEGORIES.keys())
        weights = [EMAIL_CATEGORIES[cat]["weight"] for cat in categories]
        category = random.choices(categories, weights=weights)[0]
    
    # Generate email based on category
    if category == "meeting":
        return generate_meeting_email()
    elif category == "intro":
        return generate_intro_email()
    elif category == "promotion":
        return generate_promotion_email()
    elif category == "report":
        return generate_report_email()
    elif category == "support":
        return generate_support_email()
    elif category == "security":
        return generate_security_email()
    else:
        # Default to one of the implemented categories
        implemented = ["meeting", "intro", "promotion", "report", "support", "security"]
        return generate_email(random.choice(implemented))

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