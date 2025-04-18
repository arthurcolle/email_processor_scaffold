#!/usr/bin/env python3
"""
Enhanced Email Taxonomy Generator - Creates a sophisticated hierarchical taxonomy
for email classification with 60-180 fine-grained categories.

This taxonomy is designed to handle the complexity and diversity of real-world email
communications, especially in business contexts like the Enron dataset.
"""
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Set

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enhanced_taxonomy")

# Add the project root directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import directly from the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from email_generator import EMAIL_CATEGORIES

# Define root categories for building hierarchy
ROOT_CATEGORIES = {
    # Business Operations
    "management": {
        "name": "Management",
        "description": "Management and leadership communications",
        "weight": 0.04,
    },
    "operations": {
        "name": "Operations",
        "description": "Day-to-day business operations",
        "weight": 0.04,
    },
    "project": {
        "name": "Project",
        "description": "Project-related communications",
        "weight": 0.05,
    },
    "collaboration": {
        "name": "Collaboration",
        "description": "Team collaboration and coordination",
        "weight": 0.03,
    },

    # Finance and Administration
    "finance": {
        "name": "Finance",
        "description": "Financial matters and transactions",
        "weight": 0.06,
    },
    "accounting": {
        "name": "Accounting",
        "description": "Accounting and bookkeeping",
        "weight": 0.03,
    },
    "procurement": {
        "name": "Procurement",
        "description": "Purchasing and vendor management",
        "weight": 0.02,
    },
    "admin": {
        "name": "Administrative",
        "description": "Administrative processes and communications",
        "weight": 0.02,
    },

    # Legal and Compliance
    "legal": {
        "name": "Legal",
        "description": "Legal matters and documentation",
        "weight": 0.03,
    },
    "compliance": {
        "name": "Compliance",
        "description": "Regulatory compliance and governance",
        "weight": 0.02,
    },
    "risk": {
        "name": "Risk Management",
        "description": "Risk assessment and mitigation",
        "weight": 0.02,
    },

    # Human Resources
    "hr": {
        "name": "Human Resources",
        "description": "HR communications and processes",
        "weight": 0.04,
    },
    "recruitment": {
        "name": "Recruitment",
        "description": "Hiring and recruitment processes",
        "weight": 0.03,
    },

    # IT and Technology
    "it": {
        "name": "Information Technology",
        "description": "IT systems and support",
        "weight": 0.03,
    },
    "technical": {
        "name": "Technical",
        "description": "Technical information and discussions",
        "weight": 0.03,
    },
    "system": {
        "name": "System",
        "description": "System-related communications",
        "weight": 0.03,
    },

    # Sales and Marketing
    "sales": {
        "name": "Sales",
        "description": "Sales activities and pipeline",
        "weight": 0.04,
    },
    "marketing": {
        "name": "Marketing",
        "description": "Marketing campaigns and strategies",
        "weight": 0.03,
    },
    "customer": {
        "name": "Customer",
        "description": "Customer-related communications",
        "weight": 0.04,
    },

    # Research and Development
    "research": {
        "name": "Research",
        "description": "Research and analysis",
        "weight": 0.02,
    },
    "development": {
        "name": "Development",
        "description": "Product and service development",
        "weight": 0.02,
    },
}

# Generate sub-categories for each root category
# Each tuple is (category_id, name, description, parent_id, weight)
SUB_CATEGORIES = [
    # Management categories
    ("strategic_planning", "Strategic Planning", "Long-term strategic planning and vision", "management", 0.01),
    ("executive_update", "Executive Update", "Updates and briefings for executives", "management", 0.01),
    ("board_communication", "Board Communication", "Communications with board members", "management", 0.01),
    ("performance_management", "Performance Management", "Managing and assessing performance", "management", 0.01),
    ("organizational_change", "Organizational Change", "Changes in organization structure", "management", 0.01),
    
    # Operations categories
    ("process_improvement", "Process Improvement", "Improving operational processes", "operations", 0.01),
    ("logistics", "Logistics", "Transportation and logistics management", "operations", 0.01),
    ("supply_chain", "Supply Chain", "Supply chain management", "operations", 0.01),
    ("quality_control", "Quality Control", "Quality assurance and control", "operations", 0.01),
    ("operational_issue", "Operational Issue", "Issues and challenges in operations", "operations", 0.01),
    ("business_continuity", "Business Continuity", "Ensuring continuous business operations", "operations", 0.01),
    
    # Project categories
    ("project_kickoff", "Project Kickoff", "Starting a new project", "project", 0.01),
    ("project_status", "Project Status", "Project status updates and reports", "project", 0.01),
    ("project_issue", "Project Issue", "Issues and blockers in projects", "project", 0.01),
    ("project_closure", "Project Closure", "Completing and closing projects", "project", 0.01),
    ("project_resource", "Project Resources", "Resource allocation for projects", "project", 0.01),
    ("project_schedule", "Project Schedule", "Project timelines and scheduling", "project", 0.01),
    
    # Collaboration categories
    ("team_communication", "Team Communication", "Communication within teams", "collaboration", 0.01),
    ("cross_functional", "Cross-functional Collaboration", "Collaboration across departments", "collaboration", 0.01),
    ("meeting_coordination", "Meeting Coordination", "Coordinating meetings and calls", "collaboration", 0.01),
    ("task_assignment", "Task Assignment", "Assigning tasks and responsibilities", "collaboration", 0.01),
    ("decision_making", "Decision Making", "Collaborative decision-making processes", "collaboration", 0.01),
    
    # Finance categories
    ("financial_report", "Financial Report", "Financial reporting and statements", "finance", 0.01),
    ("budget_planning", "Budget Planning", "Budget planning and forecasting", "finance", 0.01),
    ("financial_analysis", "Financial Analysis", "Analysis of financial performance", "finance", 0.01),
    ("cash_flow", "Cash Flow", "Cash flow management", "finance", 0.01),
    ("investment", "Investment", "Investment opportunities and decisions", "finance", 0.01),
    ("funding", "Funding", "Funding and capital raising", "finance", 0.01),
    ("expense_approval", "Expense Approval", "Approving business expenses", "finance", 0.01),
    
    # Accounting categories
    ("financial_statement", "Financial Statement", "Balance sheets and financial statements", "accounting", 0.01),
    ("tax_matter", "Tax Matter", "Tax-related communications", "accounting", 0.01),
    ("audit", "Audit", "Financial and operational audits", "accounting", 0.01),
    ("accounting_policy", "Accounting Policy", "Policies for accounting practices", "accounting", 0.01),
    
    # Procurement categories
    ("vendor_management", "Vendor Management", "Managing relationships with vendors", "procurement", 0.01),
    ("purchase_request", "Purchase Request", "Requests for purchases", "procurement", 0.01),
    ("supplier_negotiation", "Supplier Negotiation", "Negotiating with suppliers", "procurement", 0.01),
    ("contract_procurement", "Contract Procurement", "Procurement-related contracts", "procurement", 0.01),
    
    # Administrative categories
    ("office_management", "Office Management", "Managing office facilities", "admin", 0.01),
    ("travel_arrangement", "Travel Arrangement", "Arranging business travel", "admin", 0.01),
    ("scheduling", "Scheduling", "Scheduling and calendar management", "admin", 0.01),
    ("resource_allocation", "Resource Allocation", "Allocating office resources", "admin", 0.01),
    
    # Legal categories
    ("contract_review", "Contract Review", "Reviewing legal contracts", "legal", 0.01),
    ("legal_opinion", "Legal Opinion", "Legal advice and opinions", "legal", 0.01),
    ("litigation", "Litigation", "Legal disputes and litigation", "legal", 0.01),
    ("intellectual_property", "Intellectual Property", "Patents, trademarks, and IP", "legal", 0.01),
    ("regulatory_filing", "Regulatory Filing", "Filings with regulatory bodies", "legal", 0.01),
    
    # Compliance categories
    ("policy_compliance", "Policy Compliance", "Compliance with internal policies", "compliance", 0.01),
    ("regulatory_compliance", "Regulatory Compliance", "Compliance with external regulations", "compliance", 0.01),
    ("audit_compliance", "Audit Compliance", "Compliance-related audits", "compliance", 0.01),
    ("ethics", "Ethics", "Ethical standards and behavior", "compliance", 0.01),
    
    # Risk categories
    ("risk_assessment", "Risk Assessment", "Assessing business risks", "risk", 0.01),
    ("risk_mitigation", "Risk Mitigation", "Strategies to mitigate risks", "risk", 0.01),
    ("market_risk", "Market Risk", "Risks related to market conditions", "risk", 0.01),
    ("operational_risk", "Operational Risk", "Risks in operations", "risk", 0.01),
    
    # HR categories
    ("employee_relations", "Employee Relations", "Relationships with employees", "hr", 0.01),
    ("benefits", "Benefits", "Employee benefits administration", "hr", 0.01),
    ("compensation", "Compensation", "Employee compensation matters", "hr", 0.01),
    ("training_development", "Training & Development", "Employee training initiatives", "hr", 0.01),
    ("performance_evaluation", "Performance Evaluation", "Employee performance reviews", "hr", 0.01),
    ("policy_hr", "HR Policy", "Human resources policies", "hr", 0.01),
    ("employee_onboarding", "Employee Onboarding", "New employee onboarding", "hr", 0.01),
    ("employee_offboarding", "Employee Offboarding", "Employee departures and transitions", "hr", 0.01),
    
    # Recruitment categories
    ("job_application", "Job Application", "Job applications and inquiries", "recruitment", 0.01),
    ("candidate_evaluation", "Candidate Evaluation", "Evaluating job candidates", "recruitment", 0.01),
    ("interview_scheduling", "Interview Scheduling", "Scheduling job interviews", "recruitment", 0.01),
    ("hiring_process", "Hiring Process", "Steps in the hiring process", "recruitment", 0.01),
    ("job_offer", "Job Offer", "Extending offers to candidates", "recruitment", 0.01),
    
    # IT categories
    ("it_support", "IT Support", "Technical support and assistance", "it", 0.01),
    ("it_infrastructure", "IT Infrastructure", "IT systems and infrastructure", "it", 0.01),
    ("it_security", "IT Security", "Information security matters", "it", 0.01),
    ("it_policy", "IT Policy", "Policies for IT usage", "it", 0.01),
    ("software_license", "Software License", "Software licensing matters", "it", 0.01),
    
    # Technical categories
    ("technical_specification", "Technical Specification", "Detailed technical specifications", "technical", 0.01),
    ("technical_issue", "Technical Issue", "Problems with technical systems", "technical", 0.01),
    ("data_analysis", "Data Analysis", "Analysis of technical data", "technical", 0.01),
    ("technology_evaluation", "Technology Evaluation", "Evaluating new technologies", "technical", 0.01),
    
    # System categories
    ("system_update", "System Update", "Updates to system components", "system", 0.01),
    ("system_integration", "System Integration", "Integration between systems", "system", 0.01),
    ("system_outage", "System Outage", "System failures and outages", "system", 0.01),
    ("system_maintenance", "System Maintenance", "Maintenance of systems", "system", 0.01),
    ("system_access", "System Access", "Access rights to systems", "system", 0.01),
    
    # Sales categories
    ("sales_lead", "Sales Lead", "Potential sales opportunities", "sales", 0.01),
    ("sales_proposal", "Sales Proposal", "Proposals for potential clients", "sales", 0.01),
    ("sales_forecast", "Sales Forecast", "Forecasting future sales", "sales", 0.01),
    ("sales_pipeline", "Sales Pipeline", "Managing the sales pipeline", "sales", 0.01),
    ("competitor_analysis", "Competitor Analysis", "Analysis of competitors", "sales", 0.01),
    ("pricing_strategy", "Pricing Strategy", "Strategies for pricing products", "sales", 0.01),
    
    # Marketing categories
    ("marketing_campaign", "Marketing Campaign", "Marketing campaign planning", "marketing", 0.01),
    ("marketing_material", "Marketing Material", "Creating marketing materials", "marketing", 0.01),
    ("market_research", "Market Research", "Research on market trends", "marketing", 0.01),
    ("brand_management", "Brand Management", "Managing the company brand", "marketing", 0.01),
    ("event_marketing", "Event Marketing", "Marketing through events", "marketing", 0.01),
    
    # Customer categories
    ("customer_feedback", "Customer Feedback", "Feedback from customers", "customer", 0.01),
    ("customer_issue", "Customer Issue", "Problems raised by customers", "customer", 0.01),
    ("customer_onboarding", "Customer Onboarding", "Onboarding new customers", "customer", 0.01),
    ("customer_retention", "Customer Retention", "Retaining existing customers", "customer", 0.01),
    ("account_management", "Account Management", "Managing customer accounts", "customer", 0.01),
    
    # Research categories
    ("market_analysis", "Market Analysis", "Analysis of market conditions", "research", 0.01),
    ("research_finding", "Research Finding", "Findings from research activities", "research", 0.01),
    ("industry_trend", "Industry Trend", "Trends in the industry", "research", 0.01),
    ("competitive_intelligence", "Competitive Intelligence", "Intelligence on competitors", "research", 0.01),
    
    # Development categories
    ("product_development", "Product Development", "Developing new products", "development", 0.01),
    ("feature_request", "Feature Request", "Requests for new features", "development", 0.01),
    ("product_roadmap", "Product Roadmap", "Planning future product development", "development", 0.01),
    ("beta_testing", "Beta Testing", "Testing pre-release products", "development", 0.01),
    ("development_update", "Development Update", "Updates on development progress", "development", 0.01),

    # Energy Trading (Enron specific)
    ("energy_trading", "Energy Trading", "Trading energy commodities", "finance", 0.02),
    ("trading_strategy", "Trading Strategy", "Strategies for trading", "finance", 0.01),
    ("market_position", "Market Position", "Position in energy markets", "finance", 0.01),
    ("price_analysis", "Price Analysis", "Analysis of energy prices", "finance", 0.01),
    ("trading_risk", "Trading Risk", "Risk assessment in trading", "risk", 0.01),
    ("energy_forecast", "Energy Forecast", "Forecasting energy markets", "research", 0.01),
    ("regulatory_energy", "Energy Regulation", "Energy industry regulations", "compliance", 0.01),
]

# Add specialized categories related to Enron and energy industry
SPECIALIZED_CATEGORIES = [
    # Regulatory affairs
    ("regulatory_inquiry", "Regulatory Inquiry", "Inquiries from regulatory bodies", "compliance", 0.01),
    ("regulatory_change", "Regulatory Change", "Changes in regulations", "compliance", 0.01),
    ("regulatory_submission", "Regulatory Submission", "Submissions to regulators", "compliance", 0.01),
    
    # Trading operations
    ("trading_execution", "Trading Execution", "Executing trading operations", "operations", 0.01),
    ("trading_settlement", "Trading Settlement", "Settlement of trades", "operations", 0.01),
    ("trading_confirmation", "Trading Confirmation", "Confirmation of completed trades", "operations", 0.01),
    ("trading_limit", "Trading Limit", "Limits on trading activities", "risk", 0.01),
    
    # Energy markets
    ("power_market", "Power Market", "Electricity market information", "research", 0.01),
    ("gas_market", "Gas Market", "Natural gas market information", "research", 0.01),
    ("commodity_price", "Commodity Price", "Pricing of energy commodities", "research", 0.01),
    ("market_intelligence", "Market Intelligence", "Intelligence on energy markets", "research", 0.01),
    
    # Energy infrastructure
    ("pipeline", "Pipeline", "Pipeline operations and capacity", "operations", 0.01),
    ("transmission", "Transmission", "Power transmission systems", "operations", 0.01),
    ("storage", "Storage", "Energy storage facilities", "operations", 0.01),
    ("generation", "Generation", "Power generation assets", "operations", 0.01),
    
    # Deal structures
    ("structured_deal", "Structured Deal", "Complex deal structures", "finance", 0.01),
    ("swap_arrangement", "Swap Arrangement", "Swap transactions", "finance", 0.01),
    ("forwards_contract", "Forwards Contract", "Forward contracts", "finance", 0.01),
    ("options_trading", "Options Trading", "Trading of options", "finance", 0.01),
    
    # External relationships
    ("counterparty", "Counterparty", "Relationships with trading counterparties", "customer", 0.01),
    ("partner_communication", "Partner Communication", "Communication with business partners", "customer", 0.01),
    ("regulator_relation", "Regulator Relation", "Relationships with regulators", "compliance", 0.01),
    ("industry_group", "Industry Group", "Industry associations and groups", "external", 0.01),
]

# Add 3rd level categories for even finer granularity
GRANULAR_CATEGORIES = [
    # Project management specifics
    ("project_budget", "Project Budget", "Budget management for projects", "project_status", 0.005),
    ("project_risk", "Project Risk", "Risk management in projects", "project_status", 0.005),
    ("project_scope", "Project Scope", "Scope definition and changes", "project_status", 0.005),
    ("project_milestone", "Project Milestone", "Key project milestones", "project_status", 0.005),
    ("project_resource_allocation", "Project Resource Allocation", "Allocating resources to projects", "project_resource", 0.005),
    ("project_timeline_update", "Project Timeline Update", "Updates to project timelines", "project_schedule", 0.005),
    ("project_stakeholder_report", "Project Stakeholder Report", "Reports for project stakeholders", "project_status", 0.005),
    ("project_scope_change", "Project Scope Change", "Changes to project scope", "project_issue", 0.005),
    ("project_requirements", "Project Requirements", "Project requirements documentation", "project_kickoff", 0.005),
    ("project_quality_assurance", "Project Quality Assurance", "Quality assurance for projects", "project_status", 0.005),
    
    # Meeting specifics
    ("team_meeting", "Team Meeting", "Regular team meetings", "meeting_coordination", 0.005),
    ("client_meeting", "Client Meeting", "Meetings with clients", "meeting_coordination", 0.005),
    ("board_meeting", "Board Meeting", "Board of directors meetings", "meeting_coordination", 0.005),
    ("project_review_meeting", "Project Review Meeting", "Meetings to review project status", "meeting_coordination", 0.005),
    ("daily_standup", "Daily Standup", "Daily team synchronization meetings", "meeting_coordination", 0.005),
    ("quarterly_review", "Quarterly Review", "Quarterly business review meetings", "meeting_coordination", 0.005),
    ("executive_committee", "Executive Committee", "Executive committee meetings", "meeting_coordination", 0.005),
    ("vendor_meeting", "Vendor Meeting", "Meetings with vendors and suppliers", "meeting_coordination", 0.005),
    ("investor_meeting", "Investor Meeting", "Meetings with investors", "meeting_coordination", 0.005),
    ("emergency_meeting", "Emergency Meeting", "Urgent meetings for critical issues", "meeting_coordination", 0.005),
    
    # HR processes
    ("leave_request", "Leave Request", "Requests for time off", "employee_relations", 0.005),
    ("performance_review_process", "Performance Review Process", "Process of reviewing performance", "performance_evaluation", 0.005),
    ("compensation_adjustment", "Compensation Adjustment", "Changes to employee compensation", "compensation", 0.005),
    ("workplace_policy", "Workplace Policy", "Policies for workplace behavior", "policy_hr", 0.005),
    ("employee_grievance", "Employee Grievance", "Employee complaints and grievances", "employee_relations", 0.005),
    ("remote_work_policy", "Remote Work Policy", "Policies for remote work", "policy_hr", 0.005),
    ("leadership_development", "Leadership Development", "Leadership training and development", "training_development", 0.005),
    ("team_building", "Team Building", "Team building activities", "employee_relations", 0.005),
    ("diversity_inclusion", "Diversity & Inclusion", "Diversity and inclusion initiatives", "policy_hr", 0.005),
    ("employee_wellness", "Employee Wellness", "Employee health and wellness programs", "benefits", 0.005),
    
    # Financial reporting
    ("quarterly_financial", "Quarterly Financial", "Quarterly financial reports", "financial_report", 0.005),
    ("annual_financial", "Annual Financial", "Annual financial statements", "financial_report", 0.005),
    ("financial_forecast", "Financial Forecast", "Future financial projections", "financial_report", 0.005),
    ("profit_loss", "Profit and Loss", "Profit and loss statements", "financial_report", 0.005),
    ("balance_sheet", "Balance Sheet", "Balance sheet reports", "financial_report", 0.005),
    ("cash_flow_statement", "Cash Flow Statement", "Cash flow statements", "financial_report", 0.005),
    ("revenue_analysis", "Revenue Analysis", "Analysis of revenue streams", "financial_analysis", 0.005),
    ("expense_analysis", "Expense Analysis", "Analysis of expenses", "financial_analysis", 0.005),
    ("financial_variance", "Financial Variance", "Analysis of variance from financial plans", "financial_analysis", 0.005),
    ("cost_allocation", "Cost Allocation", "Allocation of costs across departments", "financial_analysis", 0.005),
    
    # Customer service
    ("customer_complaint", "Customer Complaint", "Complaints from customers", "customer_issue", 0.005),
    ("service_request", "Service Request", "Requests for service from customers", "customer_issue", 0.005),
    ("account_inquiry", "Account Inquiry", "Inquiries about customer accounts", "account_management", 0.005),
    ("customer_appreciation", "Customer Appreciation", "Appreciating customer relationships", "customer_retention", 0.005),
    ("product_support", "Product Support", "Support for product-related issues", "customer_issue", 0.005),
    ("service_outage", "Service Outage", "Notifications about service outages", "customer_issue", 0.005),
    ("customer_feedback_analysis", "Customer Feedback Analysis", "Analysis of customer feedback", "customer_feedback", 0.005),
    ("service_level_agreement", "Service Level Agreement", "Agreements about service levels", "account_management", 0.005),
    ("customer_satisfaction_survey", "Customer Satisfaction Survey", "Surveys measuring customer satisfaction", "customer_feedback", 0.005),
    ("case_escalation", "Case Escalation", "Escalation of customer service cases", "customer_issue", 0.005),
    
    # Legal specifics
    ("legal_dispute", "Legal Dispute", "Disputes requiring legal intervention", "litigation", 0.005),
    ("settlement_negotiation", "Settlement Negotiation", "Negotiating legal settlements", "litigation", 0.005),
    ("legal_risk_assessment", "Legal Risk Assessment", "Assessing legal risks", "legal_opinion", 0.005),
    ("confidentiality_agreement", "Confidentiality Agreement", "NDAs and confidentiality", "contract_review", 0.005),
    ("employment_law", "Employment Law", "Legal matters related to employment", "legal_opinion", 0.005),
    ("corporate_governance", "Corporate Governance", "Legal aspects of corporate governance", "legal_opinion", 0.005),
    ("trademark_application", "Trademark Application", "Applications for trademark protection", "intellectual_property", 0.005),
    ("patent_filing", "Patent Filing", "Filing for patent protection", "intellectual_property", 0.005),
    ("licensing_agreement", "Licensing Agreement", "Agreements for licensing intellectual property", "contract_review", 0.005),
    ("cease_desist", "Cease and Desist", "Demands to stop specific activities", "litigation", 0.005),
    
    # IT issues
    ("software_issue", "Software Issue", "Issues with software applications", "technical_issue", 0.005),
    ("hardware_issue", "Hardware Issue", "Issues with computer hardware", "technical_issue", 0.005),
    ("network_issue", "Network Issue", "Issues with network connectivity", "technical_issue", 0.005),
    ("data_security", "Data Security", "Security of data systems", "it_security", 0.005),
    ("database_performance", "Database Performance", "Performance issues with databases", "technical_issue", 0.005),
    ("cloud_infrastructure", "Cloud Infrastructure", "Issues with cloud-based infrastructure", "it_infrastructure", 0.005),
    ("user_access_management", "User Access Management", "Managing user access rights", "system_access", 0.005),
    ("backup_recovery", "Backup & Recovery", "Data backup and recovery processes", "it_infrastructure", 0.005),
    ("software_deployment", "Software Deployment", "Deployment of software applications", "system_update", 0.005),
    ("cybersecurity_incident", "Cybersecurity Incident", "Security incidents and breaches", "it_security", 0.005),
    
    # Marketing specifics
    ("digital_marketing", "Digital Marketing", "Marketing through digital channels", "marketing_campaign", 0.005),
    ("content_marketing", "Content Marketing", "Marketing through content creation", "marketing_campaign", 0.005),
    ("social_media_marketing", "Social Media Marketing", "Marketing on social platforms", "marketing_campaign", 0.005),
    ("campaign_results", "Campaign Results", "Results of marketing campaigns", "marketing_campaign", 0.005),
    ("email_marketing", "Email Marketing", "Marketing through email campaigns", "marketing_campaign", 0.005),
    ("seo_strategy", "SEO Strategy", "Search engine optimization strategies", "marketing_campaign", 0.005),
    ("brand_positioning", "Brand Positioning", "Positioning the brand in the market", "brand_management", 0.005),
    ("market_segmentation", "Market Segmentation", "Dividing the market into segments", "market_research", 0.005),
    ("competitor_brand_analysis", "Competitor Brand Analysis", "Analysis of competitor brands", "competitor_analysis", 0.005),
    ("marketing_roi_analysis", "Marketing ROI Analysis", "Analysis of marketing return on investment", "campaign_results", 0.005),
    
    # Sales process
    ("initial_contact", "Initial Contact", "First contact with potential clients", "sales_lead", 0.005),
    ("needs_assessment", "Needs Assessment", "Assessing client needs", "sales_lead", 0.005),
    ("proposal_preparation", "Proposal Preparation", "Preparing sales proposals", "sales_proposal", 0.005),
    ("contract_negotiation", "Contract Negotiation", "Negotiating sales contracts", "sales_proposal", 0.005),
    ("sales_objection_handling", "Sales Objection Handling", "Addressing client objections", "sales_lead", 0.005),
    ("sales_closing_techniques", "Sales Closing Techniques", "Techniques for closing sales", "sales_proposal", 0.005),
    ("customer_success_story", "Customer Success Story", "Stories of customer success", "sales_proposal", 0.005),
    ("pricing_exception", "Pricing Exception", "Exceptions to standard pricing", "pricing_strategy", 0.005),
    ("sales_territory_management", "Sales Territory Management", "Managing sales territories", "sales_pipeline", 0.005),
    ("sales_performance_analysis", "Sales Performance Analysis", "Analysis of sales performance", "sales_forecast", 0.005),
    
    # Energy trading specifics
    ("day_ahead_trading", "Day-Ahead Trading", "Trading for next-day delivery", "energy_trading", 0.005),
    ("spot_trading", "Spot Trading", "Trading in the spot market", "energy_trading", 0.005),
    ("futures_trading", "Futures Trading", "Trading future contracts", "energy_trading", 0.005),
    ("trading_position", "Trading Position", "Position in trading markets", "market_position", 0.005),
    ("options_strategy", "Options Strategy", "Strategies for trading options", "options_trading", 0.005),
    ("hedging_strategy", "Hedging Strategy", "Strategies for hedging risk", "trading_risk", 0.005),
    ("trading_algorithm", "Trading Algorithm", "Algorithms used in trading", "trading_strategy", 0.005),
    ("market_liquidity_analysis", "Market Liquidity Analysis", "Analysis of market liquidity", "market_position", 0.005),
    ("cross_commodity_spread", "Cross-Commodity Spread", "Trading based on price differences between commodities", "energy_trading", 0.005),
    ("price_volatility_analysis", "Price Volatility Analysis", "Analysis of price volatility", "price_analysis", 0.005),
    
    # Regulatory compliance specifics
    ("ferc_filing", "FERC Filing", "Filings with Federal Energy Regulatory Commission", "regulatory_submission", 0.005),
    ("sec_filing", "SEC Filing", "Filings with Securities and Exchange Commission", "regulatory_submission", 0.005),
    ("compliance_audit", "Compliance Audit", "Audits for regulatory compliance", "audit_compliance", 0.005),
    ("compliance_training", "Compliance Training", "Training on compliance matters", "policy_compliance", 0.005),
    ("data_privacy_compliance", "Data Privacy Compliance", "Compliance with data privacy regulations", "regulatory_compliance", 0.005),
    ("environmental_compliance", "Environmental Compliance", "Compliance with environmental regulations", "regulatory_compliance", 0.005),
    ("trade_compliance", "Trade Compliance", "Compliance with trade regulations", "regulatory_compliance", 0.005),
    ("aml_compliance", "AML Compliance", "Anti-money laundering compliance", "regulatory_compliance", 0.005),
    ("regulatory_investigation", "Regulatory Investigation", "Investigations by regulatory bodies", "regulatory_inquiry", 0.005),
    ("compliance_attestation", "Compliance Attestation", "Attestations of regulatory compliance", "policy_compliance", 0.005),
    
    # Additional categories for finance
    ("investment_analysis", "Investment Analysis", "Analysis of investment opportunities", "investment", 0.005),
    ("capital_expenditure", "Capital Expenditure", "Expenditures on capital assets", "budget_planning", 0.005),
    ("debt_financing", "Debt Financing", "Financing through debt instruments", "funding", 0.005),
    ("equity_financing", "Equity Financing", "Financing through equity", "funding", 0.005),
    ("tax_planning", "Tax Planning", "Planning for tax obligations", "tax_matter", 0.005),
    
    # Additional categories for operations
    ("supply_chain_disruption", "Supply Chain Disruption", "Disruptions in the supply chain", "supply_chain", 0.005),
    ("warehouse_operations", "Warehouse Operations", "Operations in warehouses", "logistics", 0.005),
    ("production_scheduling", "Production Scheduling", "Scheduling of production activities", "operations", 0.005),
    ("quality_inspection", "Quality Inspection", "Inspection for quality control", "quality_control", 0.005),
    ("capacity_planning", "Capacity Planning", "Planning for operational capacity", "operations", 0.005),
]

# Communication types (orthogonal classification)
COMMUNICATION_TYPES = [
    ("notification", "Notification", "Notification about events or changes"),
    ("request", "Request", "Request for action or information"),
    ("approval", "Approval", "Approval of actions or decisions"),
    ("announcement", "Announcement", "Public or internal announcements"),
    ("discussion", "Discussion", "Ongoing discussion or conversation"),
    ("confirmation", "Confirmation", "Confirming actions or decisions"),
    ("followup", "Follow-up", "Follow-up on previous communications"),
    ("inquiry", "Inquiry", "Questions seeking information"),
    ("response", "Response", "Responses to inquiries or requests"),
    ("update", "Update", "Updates on ongoing matters"),
    ("report", "Report", "Reporting information or results"),
    ("invitation", "Invitation", "Invitations to events or activities"),
    ("introduction", "Introduction", "Introducing people or concepts"),
    ("reminder", "Reminder", "Reminders about deadlines or events"),
    ("alert", "Alert", "Urgent notifications or warnings"),
]

# Urgency levels (orthogonal classification)
URGENCY_LEVELS = [
    ("urgent", "Urgent", "Requiring immediate attention"),
    ("high", "High Priority", "High importance, prompt attention needed"),
    ("medium", "Medium Priority", "Standard business importance"),
    ("low", "Low Priority", "For information, no immediate action required"),
    ("routine", "Routine", "Regular, recurring communication"),
]

# Function to build the enhanced taxonomy
def build_enhanced_taxonomy() -> Dict[str, Dict[str, Any]]:
    """Build the complete enhanced taxonomy for email classification"""
    taxonomy = {}
    all_categories = set()
    
    # Add root categories
    for category_id, category_info in ROOT_CATEGORIES.items():
        if category_id in all_categories:
            logger.warning(f"Duplicate category ID detected: {category_id}")
            continue
            
        all_categories.add(category_id)
        taxonomy[category_id] = {
            "name": category_info["name"],
            "description": category_info["description"],
            "weight": category_info["weight"],
            "parent": None,
            "level": 1,
            "children": []
        }
    
    # Add subcategories
    for category_id, name, description, parent_id, weight in SUB_CATEGORIES:
        if category_id in all_categories:
            logger.warning(f"Duplicate category ID detected: {category_id}")
            continue
            
        if parent_id not in taxonomy:
            logger.warning(f"Parent category not found for {category_id}: {parent_id}")
            continue
            
        all_categories.add(category_id)
        taxonomy[category_id] = {
            "name": name,
            "description": description,
            "weight": weight,
            "parent": parent_id,
            "level": 2,
            "children": []
        }
        taxonomy[parent_id]["children"].append(category_id)
    
    # Add specialized categories
    for category_id, name, description, parent_id, weight in SPECIALIZED_CATEGORIES:
        if category_id in all_categories:
            logger.warning(f"Duplicate category ID detected: {category_id}")
            continue
            
        if parent_id not in taxonomy:
            logger.warning(f"Parent category not found for {category_id}: {parent_id}")
            continue
            
        all_categories.add(category_id)
        taxonomy[category_id] = {
            "name": name,
            "description": description,
            "weight": weight,
            "parent": parent_id,
            "level": 2,
            "children": []
        }
        taxonomy[parent_id]["children"].append(category_id)
    
    # Add granular categories (3rd level)
    for category_id, name, description, parent_id, weight in GRANULAR_CATEGORIES:
        if category_id in all_categories:
            logger.warning(f"Duplicate category ID detected: {category_id}")
            continue
            
        if parent_id not in taxonomy:
            logger.warning(f"Parent category not found for {category_id}: {parent_id}")
            continue
            
        all_categories.add(category_id)
        taxonomy[category_id] = {
            "name": name,
            "description": description,
            "weight": weight,
            "parent": parent_id,
            "level": 3,
            "children": []
        }
        taxonomy[parent_id]["children"].append(category_id)
    
    # Add communication types as a separate dimension
    for category_id, name, description in COMMUNICATION_TYPES:
        comm_id = f"comm_{category_id}"
        if comm_id in all_categories:
            logger.warning(f"Duplicate category ID detected: {comm_id}")
            continue
            
        all_categories.add(comm_id)
        taxonomy[comm_id] = {
            "name": name,
            "description": description,
            "weight": 0.01,
            "parent": None,
            "level": 1,
            "dimension": "communication_type",
            "children": []
        }
    
    # Add urgency levels as a separate dimension
    for category_id, name, description in URGENCY_LEVELS:
        urgency_id = f"urgency_{category_id}"
        if urgency_id in all_categories:
            logger.warning(f"Duplicate category ID detected: {urgency_id}")
            continue
            
        all_categories.add(urgency_id)
        taxonomy[urgency_id] = {
            "name": name,
            "description": description,
            "weight": 0.01,
            "parent": None,
            "level": 1,
            "dimension": "urgency",
            "children": []
        }
    
    # Include categories from the original taxonomy to maintain compatibility
    for category_id, category_info in EMAIL_CATEGORIES.items():
        if category_id in all_categories:
            continue  # Skip if we already added this
            
        all_categories.add(category_id)
        taxonomy[category_id] = {
            "name": category_info["name"],
            "description": category_info["description"],
            "weight": category_info["weight"],
            "parent": None,
            "level": 1,
            "legacy": True,
            "children": []
        }
    
    logger.info(f"Enhanced taxonomy built with {len(taxonomy)} categories")
    return taxonomy

def get_category_count(taxonomy: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Count categories by level and dimension"""
    stats = {
        "total": len(taxonomy),
        "level_1": 0,
        "level_2": 0,
        "level_3": 0,
        "communication_types": 0,
        "urgency_levels": 0,
        "legacy": 0
    }
    
    for category_id, category in taxonomy.items():
        if category.get("legacy", False):
            stats["legacy"] += 1
        elif category.get("dimension") == "communication_type":
            stats["communication_types"] += 1
        elif category.get("dimension") == "urgency":
            stats["urgency_levels"] += 1
        elif category.get("level") == 1:
            stats["level_1"] += 1
        elif category.get("level") == 2:
            stats["level_2"] += 1
        elif category.get("level") == 3:
            stats["level_3"] += 1
    
    return stats

def get_leaf_categories(taxonomy: Dict[str, Dict[str, Any]]) -> List[str]:
    """Retrieve all leaf categories (no children)"""
    leaf_categories = []
    for category_id, category in taxonomy.items():
        if not category.get("children") and category.get("dimension") != "communication_type" and category.get("dimension") != "urgency":
            leaf_categories.append(category_id)
    return leaf_categories

def save_taxonomy(taxonomy: Dict[str, Dict[str, Any]], output_path: Optional[str] = None) -> None:
    """Save the enhanced taxonomy to a JSON file"""
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/taxonomy")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "enhanced_taxonomy.json")
    
    with open(output_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    
    logger.info(f"Enhanced taxonomy saved to {output_path}")
    
    # Save a hierarchy visualization
    hierarchy_path = os.path.join(os.path.dirname(output_path), "taxonomy_hierarchy.txt")
    with open(hierarchy_path, 'w') as f:
        f.write("# Enhanced Email Classification Taxonomy\n\n")
        
        # Write root categories first
        f.write("## Primary Categories\n\n")
        for category_id, category in sorted(taxonomy.items()):
            if category.get("level") == 1 and not category.get("dimension") and not category.get("legacy", False):
                f.write(f"- {category['name']} ({category_id}): {category['description']}\n")
                
                # Write child categories
                for child_id in sorted(category.get("children", [])):
                    if child_id in taxonomy:
                        child = taxonomy[child_id]
                        f.write(f"  - {child['name']} ({child_id}): {child['description']}\n")
                        
                        # Write grandchild categories
                        for grandchild_id in sorted(child.get("children", [])):
                            if grandchild_id in taxonomy:
                                grandchild = taxonomy[grandchild_id]
                                f.write(f"    - {grandchild['name']} ({grandchild_id}): {grandchild['description']}\n")
        
        # Write communication types
        f.write("\n## Communication Types\n\n")
        for category_id, category in sorted(taxonomy.items()):
            if category.get("dimension") == "communication_type":
                f.write(f"- {category['name']} ({category_id}): {category['description']}\n")
        
        # Write urgency levels
        f.write("\n## Urgency Levels\n\n")
        for category_id, category in sorted(taxonomy.items()):
            if category.get("dimension") == "urgency":
                f.write(f"- {category['name']} ({category_id}): {category['description']}\n")
                
        # Write legacy categories
        f.write("\n## Legacy Categories (for backward compatibility)\n\n")
        for category_id, category in sorted(taxonomy.items()):
            if category.get("legacy", False):
                f.write(f"- {category['name']} ({category_id}): {category['description']}\n")
    
    logger.info(f"Taxonomy hierarchy visualization saved to {hierarchy_path}")

if __name__ == "__main__":
    # Build and save the enhanced taxonomy
    enhanced_taxonomy = build_enhanced_taxonomy()
    
    # Print statistics
    stats = get_category_count(enhanced_taxonomy)
    logger.info("Enhanced Taxonomy Statistics:")
    logger.info(f"Total categories: {stats['total']}")
    logger.info(f"Level 1 categories: {stats['level_1']}")
    logger.info(f"Level 2 categories: {stats['level_2']}")
    logger.info(f"Level 3 categories: {stats['level_3']}")
    logger.info(f"Communication types: {stats['communication_types']}")
    logger.info(f"Urgency levels: {stats['urgency_levels']}")
    logger.info(f"Legacy categories: {stats['legacy']}")
    
    # Save to file
    save_taxonomy(enhanced_taxonomy)
    
    # Print leaf categories count
    leaf_categories = get_leaf_categories(enhanced_taxonomy)
    logger.info(f"Leaf categories (for training): {len(leaf_categories)}")