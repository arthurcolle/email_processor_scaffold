#!/usr/bin/env python3
import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
import random
from app.services.database_service import DatabaseService
from app.models.database import ExtendedClassificationType
from app.database import init_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("init_db")

# Sample data for generating random emails
SAMPLE_NAMES = [
    "Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Prince", 
    "Edward Norton", "Fiona Apple", "George Washington", "Hannah Montana", 
    "Ivan Drago", "Julia Roberts", "Kevin Hart", "Laura Palmer", 
    "Michael Scott", "Nancy Wheeler", "Oscar Martinez", "Pamela Anderson", 
    "Quentin Tarantino", "Rachel Green", "Steve Rogers", "Tina Fey"
]

SAMPLE_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", 
    "company.com", "example.org", "business.net", "school.edu",
    "acme-corp.com", "enterprise-solutions.com", "tech-innovators.io"
]

# Business email data by category
BUSINESS_EMAILS = {
    # HR Emails
    "HR": [
        {
            "subject": "Your Job Application Status - Marketing Director",
            "body": "Thank you for your application for the Marketing Director position. We're pleased to inform you that your application has been shortlisted. We would like to schedule an interview at your earliest convenience."
        },
        {
            "subject": "Employment Offer - Senior Software Engineer",
            "body": "We are pleased to offer you the position of Senior Software Engineer with Acme Corporation. This letter confirms our offer of employment with a start date of June 15, 2023. Your annual salary will be $150,000 with benefits as discussed during your interview."
        },
        {
            "subject": "Onboarding Information for Your First Day",
            "body": "Welcome to the team! We're excited to have you join us next Monday. This email contains all the information you need for your first day, including parking instructions, building access, and your orientation schedule."
        },
        {
            "subject": "Annual Performance Review - Schedule Your Session",
            "body": "It's time for your annual performance review. Please use the link below to schedule your session with your manager. Remember to complete your self-assessment form by Friday."
        },
        {
            "subject": "Important Updates to Your Employee Benefits Package",
            "body": "We're writing to inform you about several important updates to your employee benefits package that will take effect on January 1. Please review the attached documents for details about changes to our health insurance options and retirement plan."
        }
    ],
    
    # Finance Emails
    "Finance": [
        {
            "subject": "Invoice #1234 for Professional Services",
            "body": "Please find attached invoice #1234 for professional services rendered from May 1-31, 2023. Payment is due within 30 days. If you have any questions about this invoice, please contact our accounts department."
        },
        {
            "subject": "Expense Report Approval Required",
            "body": "The following expense report requires your approval: Travel Expenses - Boston Conference ($1,245.32) submitted by John Smith on June 5, 2023. Please review and approve at your earliest convenience."
        },
        {
            "subject": "Q3 Budget Planning Meeting",
            "body": "This is a reminder about our Q3 budget planning meeting scheduled for tomorrow at 2 PM in Conference Room A. Please come prepared with your department's projected expenses and revenue forecasts for the next quarter."
        },
        {
            "subject": "Monthly Financial Statement - May 2023",
            "body": "Attached is the monthly financial statement for May 2023. Key highlights include a 15% increase in revenue compared to last month and a 5% reduction in operating expenses. Please review the complete report for more details."
        },
        {
            "subject": "Important Tax Document - W-2 Available",
            "body": "Your 2022 W-2 tax document is now available in the employee portal. You will need this document to file your federal and state tax returns. Please download and save it for your records."
        }
    ],
    
    # Sales & Marketing Emails
    "Sales": [
        {
            "subject": "New Sales Lead - Enterprise Opportunity",
            "body": "I wanted to bring your attention to a new sales lead from Acme Corporation. They're interested in our enterprise solution and have requested a demo next week. This could be a significant opportunity worth approximately $250,000."
        },
        {
            "subject": "Customer Inquiry - Product Specifications",
            "body": "We've received an inquiry from a potential customer regarding the technical specifications of our latest product. They specifically want to know about integration capabilities with their existing systems. Please provide the necessary information so I can respond to them."
        },
        {
            "subject": "Price Quote Request for Johnson & Company",
            "body": "Johnson & Company has requested a formal price quotation for 500 units of our premium product line with customized branding. Please prepare a detailed quote including volume discounts and shipping costs to their Dallas headquarters."
        },
        {
            "subject": "Marketing Campaign Results - Q2 Social Media Push",
            "body": "Here are the results from our Q2 social media marketing campaign: 250,000 impressions, 15,000 clicks, 3,500 leads generated, and 750 conversions. The ROI was 315%, significantly exceeding our target of 200%. Recommendations for Q3 are attached."
        },
        {
            "subject": "Competitive Analysis Report - New Market Entrant",
            "body": "Attached is an analysis of the new competitor that entered our market last month. The report includes their pricing strategy, product features, target customer segments, and perceived strengths and weaknesses compared to our offerings."
        }
    ],
    
    # Legal & Compliance Emails
    "Legal": [
        {
            "subject": "Contract Review Request - Vendor Agreement",
            "body": "Please review the attached vendor agreement with ABC Suppliers. We need to ensure all terms are favorable and compliant with our procurement policies. Please provide your feedback by Friday so we can proceed with the signing."
        },
        {
            "subject": "Legal Notice - Trademark Infringement",
            "body": "We have identified a potential trademark infringement by XYZ Corporation, who is using branding elements similar to our registered trademark. We recommend sending a cease and desist letter. Please review the attached documentation and advise."
        },
        {
            "subject": "Quarterly Compliance Report Due",
            "body": "This is a reminder that your department's quarterly compliance report is due by the end of this week. Please ensure all incidents are properly documented and all staff have completed their required compliance training."
        },
        {
            "subject": "Confidential - NDA with Strategic Partner",
            "body": "Attached is the non-disclosure agreement for our upcoming strategic partnership with Innovation Tech. Please review and sign before our joint development meetings begin next week. All discussions should be considered confidential."
        },
        {
            "subject": "GDPR Data Subject Request",
            "body": "We've received a data subject access request from a customer in the EU. Under GDPR, we are required to provide all personal data we hold about this individual within 30 days. Please help identify any relevant data in your department's systems."
        }
    ],
    
    # IT & Operations Emails
    "IT": [
        {
            "subject": "IT Support Ticket #4567 - Email Access Issue",
            "body": "We've received your support ticket #4567 regarding email access issues. Our technician will contact you within the next hour to troubleshoot the problem. In the meantime, please try clearing your browser cache and restarting your computer."
        },
        {
            "subject": "URGENT: System Alert - Database Server",
            "body": "ALERT: The primary database server is currently experiencing high CPU usage (95%) and decreased response times. The IT team is investigating the issue. Please minimize database-intensive operations until further notice."
        },
        {
            "subject": "Scheduled Maintenance - Company Portal",
            "body": "The company portal will be unavailable due to scheduled maintenance this Saturday from 10 PM to 2 AM. During this time, we will be upgrading to the latest version and implementing security patches. No action is required on your part."
        },
        {
            "subject": "Software License Renewal - Adobe Creative Suite",
            "body": "Your department's Adobe Creative Suite licenses are due for renewal on July 15. The annual cost is $15,000 for 10 user licenses. Please confirm by replying to this email whether you wish to renew, upgrade, or adjust the number of licenses."
        },
        {
            "subject": "Production Issue Resolution - Order Processing System",
            "body": "This is to inform you that the order processing system issue reported earlier today has been resolved. The root cause was identified as a database connection timeout. All pending orders have been processed, and the system is now functioning normally."
        }
    ]
}

# General email subjects
SAMPLE_SUBJECTS = [
    "Weekly Team Update", "Important Project Deadline", "Meeting Invitation",
    "Follow-up on Yesterday's Discussion", "New Product Launch",
    "Holiday Schedule", "Quarterly Report", "Training Opportunity",
    "Reminder: Submission Due Today", "Company Announcement",
    "Customer Feedback Summary", "Request for Information",
    "Happy Birthday!", "Office Closure Notice", "Welcome to the Team",
    "Policy Update", "Invitation to Company Event", "Security Alert",
    "Survey Response Needed", "Thank You for Your Support"
]

# General email bodies
SAMPLE_BODIES = [
    "Just wanted to follow up on our previous conversation. Let me know if you have any questions.",
    "Please find attached the report we discussed in yesterday's meeting. Your feedback is appreciated.",
    "I'm writing to invite you to our annual company picnic next Friday at 3pm in Central Park.",
    "This is a reminder that your quarterly tax payment is due by the end of this week.",
    "We're excited to announce that our new product line will be launching next month!",
    "Thank you for your recent purchase. We hope you're enjoying your new product.",
    "We're having a flash sale this weekend with 50% off all items. Don't miss out!",
    "Your flight to Paris has been confirmed. Departure: 9:30 AM, Terminal B.",
    "Please review the attached document and provide your signature by Friday.",
    "We've updated our privacy policy. Please review the changes at your earliest convenience.",
    "This is a notification that your account was accessed from a new device.",
    "Your package has been shipped and is expected to arrive on Wednesday.",
    "Happy birthday! As a gift, please enjoy this 20% discount on your next purchase.",
    "We're sorry to hear about your recent experience. We'd like to offer a refund.",
    "Your subscription will renew automatically on June 15. Please update your payment method."
]

def generate_random_email():
    """Generate random email data for testing"""
    # Decide whether to generate a general email or business-specific email
    is_business_email = random.random() > 0.4  # 60% chance of a business email
    
    if is_business_email:
        # Select a business category
        categories = list(BUSINESS_EMAILS.keys())
        category = random.choice(categories)
        
        # Select a template from that category
        template = random.choice(BUSINESS_EMAILS[category])
        subject = template["subject"]
        body = template["body"]
        
        # For business emails, use a business domain for the sender
        business_domains = ["acme-corp.com", "enterprise-solutions.com", "tech-innovators.io", 
                           "business.net", "company.com", "example.org"]
        sender_name = random.choice(SAMPLE_NAMES)
        sender_domain = random.choice(business_domains)
        sender_email = f"{sender_name.lower().replace(' ', '.')}@{sender_domain}"
        
        # Add business-specific labels
        available_labels = [category.lower(), "important", "work", "business", "priority"]
    else:
        # Generate a general email
        sender_name = random.choice(SAMPLE_NAMES)
        sender_domain = random.choice(SAMPLE_DOMAINS)
        sender_email = f"{sender_name.lower().replace(' ', '.')}@{sender_domain}"
        
        # Subject and body
        subject = random.choice(SAMPLE_SUBJECTS)
        body = random.choice(SAMPLE_BODIES)
        
        # General labels
        available_labels = ["personal", "urgent", "follow-up", "finance", "travel", "shopping", "social"]
    
    # Generate recipients (1-3)
    recipient_count = random.randint(1, 3)
    recipients = []
    
    for _ in range(recipient_count):
        name = random.choice(SAMPLE_NAMES)
        email = f"{name.lower().replace(' ', '.')}@{random.choice(SAMPLE_DOMAINS)}"
        recipients.append({"email": email, "name": name})
    
    # Generate CC recipients (0-2)
    cc_count = random.randint(0, 2)
    cc = []
    
    for _ in range(cc_count):
        name = random.choice(SAMPLE_NAMES)
        email = f"{name.lower().replace(' ', '.')}@{random.choice(SAMPLE_DOMAINS)}"
        cc.append({"email": email, "name": name})
    
    # Random received date (within last 30 days)
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)
    received_at = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
    
    # Random state flags - business emails are more likely to be important
    if is_business_email:
        read = random.random() > 0.2  # 80% chance of being read
        starred = random.random() > 0.6  # 40% chance of being starred
        important = random.random() > 0.4  # 60% chance of being important
    else:
        read = random.random() > 0.3  # 70% chance of being read
        starred = random.random() > 0.8  # 20% chance of being starred
        important = random.random() > 0.7  # 30% chance of being important
    
    # Random labels (0-3)
    label_count = random.randint(1, 3) if is_business_email else random.randint(0, 2)
    if label_count > len(available_labels):
        label_count = len(available_labels)
    labels = random.sample(available_labels, label_count)
    
    # Construct email data
    email_data = {
        "sender": {"email": sender_email, "name": sender_name},
        "recipients": recipients,
        "cc": cc,
        "subject": subject,
        "body": body,
        "received_at": received_at,
        "read": read,
        "starred": starred,
        "important": important,
        "labels": labels
    }
    
    return email_data

def get_classification_for_email(email_data):
    """Determine a classification for an email based on its content"""
    subject = email_data.get("subject", "").lower()
    body = email_data.get("body", "").lower()
    combined = subject + " " + body
    
    # Define keywords for each classification
    classification_keywords = {
        # Original and basic classifications
        ExtendedClassificationType.MEETING: ["meeting", "discussion", "appointment", "calendar", "schedule", "zoom", "teams", "call", "conference"],
        ExtendedClassificationType.INTRO: ["introduction", "hello", "hi there", "nice to meet", "introduce", "introducing", "new team member"],
        ExtendedClassificationType.PROMOTION: ["sale", "discount", "offer", "promo", "off", "deal", "limited time", "exclusive", "savings", "special offer"],
        ExtendedClassificationType.NEWSLETTER: ["newsletter", "update", "weekly", "monthly", "digest", "latest news", "roundup", "bulletin"],
        ExtendedClassificationType.NOTIFICATION: ["notification", "alert", "reminder", "notice", "update", "fyi", "action required"],
        ExtendedClassificationType.TRANSACTION: ["transaction", "purchase", "receipt", "order", "payment confirmation"],
        ExtendedClassificationType.INVITATION: ["invitation", "invite", "join", "rsvp", "attend", "participate", "event"],
        ExtendedClassificationType.ALERT: ["alert", "warning", "caution", "important notice", "attention", "urgent", "critical"],
        ExtendedClassificationType.ANNOUNCEMENT: ["announcement", "announcing", "we're excited", "proud to present", "new release"],
        ExtendedClassificationType.RECEIPT: ["receipt", "your purchase", "order confirmation", "thank you for your purchase"],
        ExtendedClassificationType.REQUEST: ["request", "asking", "please provide", "need your", "could you", "action needed"],
        ExtendedClassificationType.SOCIAL: ["social", "friend", "connection", "network", "community", "linkedin", "facebook"],
        ExtendedClassificationType.UPDATES: ["update", "change", "new version", "upgrade", "improvement", "enhancement"],
        ExtendedClassificationType.SECURITY: ["security", "password", "login", "access", "authentication", "2fa", "verify"],
        ExtendedClassificationType.PERSONAL: ["personal", "private", "confidential", "between us", "just for you"],
        ExtendedClassificationType.BILL: ["bill", "due", "payment", "balance", "account", "invoice due"],
        ExtendedClassificationType.TRAVEL: ["travel", "flight", "hotel", "reservation", "booking", "trip", "itinerary"],
        ExtendedClassificationType.SHIPPING: ["shipping", "delivery", "package", "shipment", "track", "order shipped"],
        ExtendedClassificationType.SURVEY: ["survey", "feedback", "opinion", "review", "rate", "questionnaire"],
        
        # HR Classifications
        ExtendedClassificationType.JOB_APPLICATION: ["job application", "application status", "resume", "cover letter", "applicant", "position", "job opportunity"],
        ExtendedClassificationType.EMPLOYMENT_OFFER: ["job offer", "offer letter", "employment contract", "compensation", "start date", "welcome aboard", "position offer"],
        ExtendedClassificationType.ONBOARDING: ["onboarding", "first day", "orientation", "new hire", "welcome pack", "employee setup", "getting started"],
        ExtendedClassificationType.PERFORMANCE_REVIEW: ["performance review", "evaluation", "feedback", "goals", "objectives", "assessment", "annual review", "quarterly review"],
        ExtendedClassificationType.BENEFITS: ["benefits", "health insurance", "retirement", "401k", "pension", "dental", "vision", "open enrollment", "hr benefits"],
        ExtendedClassificationType.PAYROLL: ["payroll", "salary", "direct deposit", "pay stub", "compensation", "tax withholding", "payment", "wages"],
        ExtendedClassificationType.TRAINING: ["training", "course", "workshop", "learning", "development", "certification", "skills", "educational", "professional development"],
        ExtendedClassificationType.RECRUITMENT: ["recruitment", "hiring", "talent acquisition", "candidate", "interview", "referral", "job posting", "recruitment drive"],
        ExtendedClassificationType.EMPLOYEE_ENGAGEMENT: ["employee engagement", "satisfaction survey", "team building", "company culture", "morale", "pulse survey", "recognition"],
        ExtendedClassificationType.EMPLOYEE_OFFBOARDING: ["offboarding", "exit interview", "resignation", "termination", "last day", "farewell", "departure", "leaving"],
        
        # Finance Classifications
        ExtendedClassificationType.INVOICE: ["invoice", "bill to", "payment due", "billing", "net terms", "invoice number", "amount due", "please pay"],
        ExtendedClassificationType.EXPENSE_REPORT: ["expense report", "reimbursement", "business expense", "receipt submission", "travel expense", "expense claim"],
        ExtendedClassificationType.BUDGET: ["budget", "forecast", "financial plan", "allocation", "spend", "cost center", "fiscal year", "quarterly budget"],
        ExtendedClassificationType.FINANCIAL_STATEMENT: ["financial statement", "balance sheet", "income statement", "cash flow", "profit and loss", "quarterly results"],
        ExtendedClassificationType.TAX_DOCUMENT: ["tax document", "w2", "1099", "tax return", "tax filing", "irs", "tax form", "tax preparation", "deduction"],
        ExtendedClassificationType.PURCHASE_ORDER: ["purchase order", "po number", "order form", "procurement", "vendor order", "authorized purchase"],
        ExtendedClassificationType.PAYMENT_CONFIRMATION: ["payment confirmation", "transaction complete", "payment received", "payment processed", "successfully paid"],
        ExtendedClassificationType.CREDIT_MEMO: ["credit memo", "credit note", "refund", "account credit", "adjustment", "balance adjustment"],
        ExtendedClassificationType.AUDIT: ["audit", "financial review", "compliance check", "account audit", "internal audit", "audit findings"],
        ExtendedClassificationType.FUNDING: ["funding", "investment", "venture capital", "financing", "capital raise", "fundraising", "seed round", "series a"],
        
        # Sales & Marketing Classifications
        ExtendedClassificationType.LEAD: ["sales lead", "prospect", "potential customer", "interested client", "lead generation", "sales opportunity"],
        ExtendedClassificationType.CUSTOMER_INQUIRY: ["customer inquiry", "product question", "service inquiry", "customer asking", "request information"],
        ExtendedClassificationType.QUOTATION: ["quotation", "price quote", "estimate", "proposed cost", "pricing", "quote request", "proposal", "bid"],
        ExtendedClassificationType.SALES_OPPORTUNITY: ["sales opportunity", "potential deal", "business opportunity", "prospective sale", "deal pipeline"],
        ExtendedClassificationType.MARKETING_CAMPAIGN: ["marketing campaign", "campaign launch", "advertising", "promotion strategy", "campaign results", "marketing initiative"],
        ExtendedClassificationType.CUSTOMER_FEEDBACK: ["customer feedback", "client response", "product review", "service feedback", "customer opinion", "testimonial"],
        ExtendedClassificationType.COMPETITOR_ANALYSIS: ["competitor analysis", "competitive landscape", "market competition", "competitor research", "industry analysis"],
        ExtendedClassificationType.MARKET_RESEARCH: ["market research", "industry trends", "customer insights", "market data", "focus group", "research findings"],
        ExtendedClassificationType.PRESS_RELEASE: ["press release", "media announcement", "news release", "public relations", "media relations", "news announcement"],
        ExtendedClassificationType.BRAND_ASSETS: ["brand assets", "logo files", "brand guidelines", "style guide", "visual identity", "brand materials"],
        
        # Legal & Compliance Classifications
        ExtendedClassificationType.CONTRACT: ["contract", "agreement", "terms", "legal document", "signed contract", "contractual", "legal agreement"],
        ExtendedClassificationType.LEGAL_NOTICE: ["legal notice", "formal notice", "cease and desist", "legal communication", "official notice", "legal correspondence"],
        ExtendedClassificationType.COMPLIANCE: ["compliance", "regulatory", "industry standards", "requirements", "compliance update", "compliance report", "adherence"],
        ExtendedClassificationType.REGULATORY: ["regulatory", "regulation", "regulator", "compliance requirement", "regulatory change", "regulatory authority"],
        ExtendedClassificationType.NDA: ["nda", "non-disclosure", "confidentiality agreement", "confidentiality", "proprietary information", "confidential"],
        ExtendedClassificationType.INTELLECTUAL_PROPERTY: ["intellectual property", "patent", "trademark", "copyright", "ip rights", "ip protection", "ip filing"],
        ExtendedClassificationType.LITIGATION: ["litigation", "lawsuit", "legal action", "legal proceedings", "court case", "legal dispute", "settlement"],
        ExtendedClassificationType.GDPR: ["gdpr", "data protection", "privacy law", "eu regulation", "data subject", "privacy compliance", "data rights"],
        ExtendedClassificationType.DATA_PRIVACY: ["data privacy", "privacy policy", "personal data", "information privacy", "privacy notice", "data handling"],
        ExtendedClassificationType.TERMS_CONDITIONS: ["terms and conditions", "terms of use", "terms of service", "user agreement", "legal terms", "service terms"],
        
        # IT & Operations Classifications
        ExtendedClassificationType.IT_SUPPORT: ["it support", "technical help", "helpdesk", "tech support", "it assistance", "technical issue", "it ticket"],
        ExtendedClassificationType.SYSTEM_ALERT: ["system alert", "server notification", "system notification", "outage alert", "service alert", "system warning"],
        ExtendedClassificationType.MAINTENANCE_NOTIFICATION: ["maintenance", "scheduled maintenance", "system upgrade", "downtime", "maintenance window"],
        ExtendedClassificationType.SOFTWARE_LICENSE: ["software license", "license key", "activation", "subscription", "license renewal", "license agreement"],
        ExtendedClassificationType.DATA_BACKUP: ["data backup", "backup complete", "backup failed", "recovery", "data protection", "system backup"],
        ExtendedClassificationType.INCIDENT_REPORT: ["incident report", "incident notification", "issue report", "problem report", "service incident"],
        ExtendedClassificationType.PROJECT_STATUS: ["project status", "milestone update", "project progress", "status report", "project timeline", "deliverables"],
        ExtendedClassificationType.PRODUCTION_ISSUE: ["production issue", "service disruption", "system failure", "production error", "critical bug", "outage"],
        ExtendedClassificationType.VENDOR_COMMUNICATION: ["vendor", "supplier", "third-party", "partner communication", "vendor update", "service provider"],
        ExtendedClassificationType.SERVICE_LEVEL_AGREEMENT: ["sla", "service level", "agreement terms", "service performance", "service guarantee", "service metrics"]
    }
    
    # Check for matches and pick the classification with the most matches
    matches = {}
    for classification, keywords in classification_keywords.items():
        match_count = sum(1 for keyword in keywords if keyword in combined)
        if match_count > 0:
            matches[classification] = match_count
    
    # If we found matches, return the one with the most matches
    if matches:
        best_match = max(matches.items(), key=lambda x: x[1])[0]
        return best_match.value
    
    # Default to UNKNOWN if no matches
    return ExtendedClassificationType.UNKNOWN.value

async def initialize_database():
    """Initialize the database and populate it with sample data"""
    logger.info("Initializing database...")
    
    # Initialize database tables
    init_db()
    
    # Create database service
    db_service = DatabaseService()
    
    # Generate and insert sample emails
    email_count = 100
    logger.info(f"Generating {email_count} sample emails...")
    
    for i in range(email_count):
        # Generate random email data
        email_data = generate_random_email()
        
        # Create email in database
        email_id = db_service.create_email(email_data)
        
        # Classify email
        classification = get_classification_for_email(email_data)
        confidence = random.uniform(0.7, 0.99)
        db_service.classify_email(email_id, classification, confidence, "model-v1.0")
        
        if (i + 1) % 10 == 0:
            logger.info(f"Created {i + 1}/{email_count} emails")
    
    logger.info("Database initialization complete!")
    logger.info("Sample emails have been created with various classifications")

if __name__ == "__main__":
    asyncio.run(initialize_database())