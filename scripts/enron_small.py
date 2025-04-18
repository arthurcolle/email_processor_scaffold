#!/usr/bin/env python3
"""
Small Enron Email sample classifier for demonstration purposes.
This script uses a pre-defined set of Enron emails to test our classifier without
needing to download the full dataset.
"""
import os
import sys
import logging
import json
import time

# Add parent directory to path for importing our classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_email import predict_email_category

# Initialize the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enron_small")

# Sample Enron emails (a small representative set)
SAMPLE_EMAILS = [
    {
        "subject": "Schedule for visit",
        "body": "Hello Ken,\n\nAttached is your schedule for your visit to New York. Please let me know if you have any questions.\n\nRegards,\nSherri Sera\nExecutive Assistant to Kenneth Lay\nOffice: (713) 853-5984",
        "sender": "sherri.sera@enron.com",
        "recipients": "kenneth.lay@enron.com"
    },
    {
        "subject": "FW: Website design feedback needed",
        "body": "I think this website gives a good picture of one of our competitors. Can we look at doing something like this for our group?\n\n--John\n\n-----Original Message-----\nFrom: Marketing Dept\nSent: Friday, May 12, 2000 10:36 AM\nTo: All Employees\nSubject: Website design feedback needed\n\nWe're gathering feedback on our new website design. Please visit www.example.com and send your comments to marketing@enron.com.",
        "sender": "john.doe@enron.com",
        "recipients": "development.team@enron.com"
    },
    {
        "subject": "Report: April Trading Activity",
        "body": "Attached is the monthly trading activity report for April 2001.\n\nHighlights:\n- Total volume: 1.2M MMBtu\n- Average price: $5.42\n- P&L: $3.2M\n\nLet me know if you need any clarification on the numbers.\n\nRegards,\nSara",
        "sender": "sara.shackleton@enron.com",
        "recipients": "trading.desk@enron.com"
    },
    {
        "subject": "RE: Meeting tomorrow canceled",
        "body": "Thanks for letting me know. Let's reschedule for next week. I'm available Tuesday afternoon or Wednesday morning.\n\nJeff",
        "sender": "jeff.skilling@enron.com",
        "recipients": "louise.kitchen@enron.com"
    },
    {
        "subject": "Urgent: System maintenance tonight",
        "body": "All,\n\nThe email system will be down for maintenance tonight from 10pm to 2am CDT. Please save any work in progress before leaving today.\n\nIT Support Team",
        "sender": "it.support@enron.com",
        "recipients": "all.employees@enron.com"
    },
    {
        "subject": "Legal Review: Johnson Contract",
        "body": "Andy,\n\nI've reviewed the Johnson contract and have a few concerns about section 3.2 regarding liability limitations. Can we schedule some time tomorrow to discuss?\n\nStacey Akin\nLegal Department\nEnron Corp.\n(713) 853-3967",
        "sender": "stacey.akin@enron.com",
        "recipients": "andy.zipper@enron.com"
    },
    {
        "subject": "Reminder: Quarterly performance reviews due",
        "body": "Managers,\n\nThis is a reminder that Q2 performance reviews are due this Friday. Please complete all outstanding reviews by 5pm.\n\nHR Department",
        "sender": "hr@enron.com",
        "recipients": "all.managers@enron.com"
    },
    {
        "subject": "California situation update",
        "body": "Team,\n\nAttached is the latest analysis of the California energy crisis and how it affects our positions. The regulatory team is closely monitoring developments and will provide daily updates.\n\nKey points:\n- FERC meeting scheduled for Friday\n- Price caps being discussed\n- Our exposure is approximately $25M\n\nPlease keep this information confidential.\n\nRegards,\nRichard",
        "sender": "richard.sanders@enron.com",
        "recipients": "legal.team@enron.com"
    },
    {
        "subject": "Expense report approval needed",
        "body": "Please approve my expense report for the Chicago trip (ID: ER29384). Total amount: $1,542.68.\n\nMain expenses:\n- Airfare: $425\n- Hotel (3 nights): $684\n- Client dinner: $320\n\nThanks,\nMike",
        "sender": "mike.grigsby@enron.com",
        "recipients": "vince.kaminski@enron.com"
    },
    {
        "subject": "Holiday Party - December 15",
        "body": "You're invited to the annual Enron holiday party!\n\nDate: December 15, 2000\nTime: 7:00 PM - 11:00 PM\nLocation: Hyatt Regency Downtown\nDress: Formal\n\nPlease RSVP by December 8th.\n\nHope to see you there!\n\nEvent Committee",
        "sender": "events@enron.com",
        "recipients": "all.employees@enron.com"
    },
    {
        "subject": "Your password will expire in 3 days",
        "body": "Your network password will expire in 3 days.\n\nPlease change your password by visiting the IT portal or pressing Ctrl+Alt+Del and selecting 'Change Password'.\n\nRemember that passwords must be at least 8 characters and include a mix of letters, numbers, and symbols.\n\nIT Security Team",
        "sender": "security@enron.com",
        "recipients": "louise.kitchen@enron.com"
    },
    {
        "subject": "Energy derivatives article",
        "body": "I thought you might find this article interesting. It discusses the growth of energy derivatives and mentions our leadership in the space.\n\nLet me know your thoughts.\n\nhttp://www.energytrading.com/articles/derivatives-growth\n\nRegards,\nJeff",
        "sender": "jeff.skilling@enron.com",
        "recipients": "kenneth.lay@enron.com"
    },
    {
        "subject": "Resume: Potential Trading Assistant",
        "body": "Hello,\n\nI applied for the Trading Assistant position last week (Job ID: 458293) and wanted to follow up. I have 3 years of experience in energy trading operations at Reliant and believe I would be a great fit for your team.\n\nMy resume is attached for your review. I'm available for an interview anytime next week.\n\nThank you for your consideration.\n\nMichael Johnson\n(713) 555-1234",
        "sender": "michael.johnson@gmail.com",
        "recipients": "recruiting@enron.com"
    },
    {
        "subject": "Enron stock trading at $68.50",
        "body": "FYI - Enron closed at $68.50 today, up 2.5%. The analyst call went well, and Goldman Sachs has upgraded us to a 'Strong Buy' rating.\n\nThe stock has now outperformed the S&P 500 by 25% year-to-date.\n\nInvestor Relations",
        "sender": "ir@enron.com",
        "recipients": "executive.committee@enron.com"
    },
    {
        "subject": "New benefit: Enron Employee Stock Purchase Plan",
        "body": "We're pleased to announce a new benefit for all Enron employees - the Employee Stock Purchase Plan (ESPP).\n\nHighlights:\n- Purchase Enron stock at a 15% discount\n- Contribute up to 10% of your salary\n- Quarterly purchase periods\n\nEnrollment begins next Monday. See the attached brochure for details.\n\nHR Benefits Team",
        "sender": "benefits@enron.com",
        "recipients": "all.employees@enron.com"
    }
]

def classify_sample_emails():
    """Classify the sample Enron emails using our classifier"""
    logger.info(f"Classifying {len(SAMPLE_EMAILS)} sample Enron emails...")
    
    results = []
    start_time = time.time()
    
    for i, email_data in enumerate(SAMPLE_EMAILS):
        # Get classification
        category, confidence = predict_email_category(
            email_data['subject'], 
            email_data['body']
        )
        
        # Add classification to results
        result = {
            'subject': email_data['subject'],
            'category': category,
            'confidence': confidence,
            'sender': email_data.get('sender', ''),
            'recipients': email_data.get('recipients', '')
        }
        results.append(result)
        
        # Print result
        logger.info(f"Email {i+1}: \"{email_data['subject']}\"")
        logger.info(f"  → Category: {category} (Confidence: {confidence:.2f})")
    
    # Calculate statistics
    categories = {}
    total_confidence = 0.0
    
    for result in results:
        category = result['category']
        confidence = result['confidence']
        
        if category not in categories:
            categories[category] = {
                'count': 0,
                'total_confidence': 0.0
            }
        
        categories[category]['count'] += 1
        categories[category]['total_confidence'] += confidence
        total_confidence += confidence
    
    # Calculate averages and percentages
    avg_confidence = total_confidence / len(results) if results else 0
    
    for category in categories:
        count = categories[category]['count']
        categories[category]['percentage'] = (count / len(results)) * 100
        categories[category]['avg_confidence'] = categories[category]['total_confidence'] / count
    
    # Sort categories by count
    sorted_categories = sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Save the results
    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "data/enron/sample_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'statistics': {
                'total_emails': len(results),
                'average_confidence': avg_confidence,
                'execution_time_seconds': time.time() - start_time,
                'category_distribution': {
                    cat: {
                        'count': data['count'],
                        'percentage': data['percentage'],
                        'avg_confidence': data['avg_confidence']
                    } for cat, data in sorted_categories
                }
            }
        }, f, indent=2)
    
    # Print summary statistics
    logger.info("\nClassification Summary:")
    logger.info(f"Total emails classified: {len(results)}")
    logger.info(f"Average confidence: {avg_confidence:.2f}")
    logger.info(f"Execution time: {time.time() - start_time:.2f} seconds")
    logger.info("\nCategory Distribution:")
    
    for category, data in sorted_categories:
        logger.info(f"  {category}: {data['count']} emails ({data['percentage']:.1f}%), " +
                  f"avg confidence: {data['avg_confidence']:.2f}")
    
    logger.info(f"\nResults saved to {results_file}")
    
    return results

if __name__ == "__main__":
    classify_sample_emails()